# https://github.com/pencil-code/pencil-code

```console
python/pencil/visu/pv_volume_plotter.py:and works on that. In my case I have a dedicated NVIDIA GPU - with different system
python/pencil/visu/pv_volume_plotter.py:                # Possible mappers for add_volume are: 'fixed_point', 'gpu',
config/compilers/separate/nvidia-fortran.conf:	FFLAGS += -O -cuda
config/hosts/grsarson/langkawi.conf:# langkawi (gpu machine in newcastle)
config/hosts/grsarson/langkawi.conf:  LDFLAGS_HELPER += -lcuda -lcudart
config/hosts/grsarson/langkawi.conf:  GPULIB += -lcuda -lcudart
.gitattributes:config/compilers/separate/nvidia-fortran.conf -text
.gitattributes:samples/gputest/README -text
.gitattributes:samples/gputest/job -text
.gitattributes:samples/gputest/k.dat -text
.gitattributes:samples/gputest/print.in -text
.gitattributes:samples/gputest/run.in -text
.gitattributes:samples/gputest/src/Makefile.local -text
.gitattributes:samples/gputest/src/cparam.local -text
.gitattributes:samples/gputest/start.in -text
.gitattributes:samples/gputest/video.in -text
.gitattributes:src/astaroth/gpu_astaroth.cc -text
.gitattributes:src/cuda/gpu_astaroth.cu -text
.gitattributes:src/cuda/src/TODO.txt -text
.gitattributes:src/cuda/src/ac_run_gdb -text
.gitattributes:src/cuda/src/alctest.cu -text
.gitattributes:src/cuda/src/alcyone_template.sh -text
.gitattributes:src/cuda/src/alcyone_template_memcheck.sh -text
.gitattributes:src/cuda/src/analysis/python/README -text
.gitattributes:src/cuda/src/analysis/python/anim.py -text
.gitattributes:src/cuda/src/analysis/python/animate_data.py -text
.gitattributes:src/cuda/src/analysis/python/compare3.py -text
.gitattributes:src/cuda/src/analysis/python/compare_data.py -text
.gitattributes:src/cuda/src/analysis/python/compare_gausxyz_plane.py -text
.gitattributes:src/cuda/src/analysis/python/derivs.py -text
.gitattributes:src/cuda/src/analysis/python/plotting_tools.py -text
.gitattributes:src/cuda/src/analysis/python/powerspectrum.py -text
.gitattributes:src/cuda/src/analysis/python/read_k_used.py -text
.gitattributes:src/cuda/src/analysis/python/read_k_used_view.py -text
.gitattributes:src/cuda/src/analysis/python/read_snapshot.py -text
.gitattributes:src/cuda/src/analysis/python/read_ts.py -text
.gitattributes:src/cuda/src/analysis/python/read_ts_show.py -text
.gitattributes:src/cuda/src/analysis/python/visual_data.py -text
.gitattributes:src/cuda/src/analysis/python/vtk_convert.py -text
.gitattributes:src/cuda/src/analysis/python/yt_visual.py -text
.gitattributes:src/cuda/src/analysis/random_velocity.gif -text
.gitattributes:src/cuda/src/anim.py -text
.gitattributes:src/cuda/src/animate_data.py -text
.gitattributes:src/cuda/src/animation_images/README -text
.gitattributes:src/cuda/src/astaroth_sgl.so -text
.gitattributes:src/cuda/src/boundcond.cu -text
.gitattributes:src/cuda/src/boundcond.cuh -text
.gitattributes:src/cuda/src/collectiveops.cu -text
.gitattributes:src/cuda/src/collectiveops.cuh -text
.gitattributes:src/cuda/src/compare3.py -text
.gitattributes:src/cuda/src/compare_data.py -text
.gitattributes:src/cuda/src/compare_gausxyz_plane.py -text
.gitattributes:src/cuda/src/compiler_out.txt -text
.gitattributes:src/cuda/src/compute.cu -text
.gitattributes:src/cuda/src/continuity.cu -text
.gitattributes:src/cuda/src/continuity.cuh -text
.gitattributes:src/cuda/src/copyHalosConcur.cu -text
.gitattributes:src/cuda/src/copyHalosConcur.cuh -text
.gitattributes:src/cuda/src/copy_halos.cu -text
.gitattributes:src/cuda/src/copyhalos.cuh -text
.gitattributes:src/cuda/src/copyinternalhalostohost.cu -text
.gitattributes:src/cuda/src/copyouterhalostodevice.cu -text
.gitattributes:src/cuda/src/coriolis.cu -text
.gitattributes:src/cuda/src/coriolis.cuh -text
.gitattributes:src/cuda/src/data/README -text
.gitattributes:src/cuda/src/data/animation/README -text
.gitattributes:src/cuda/src/dconsts.cuh -text
.gitattributes:src/cuda/src/dconstsextern.cuh -text
.gitattributes:src/cuda/src/ddiagsextern.cuh -text
.gitattributes:src/cuda/src/defines.h -text
.gitattributes:src/cuda/src/defines_PC.h -text
.gitattributes:src/cuda/src/defines_dims_PC.h -text
.gitattributes:src/cuda/src/derivs.py -text
.gitattributes:src/cuda/src/dfdf.cuh -text
.gitattributes:src/cuda/src/dfdfextern.cuh -text
.gitattributes:src/cuda/src/diagnostics.cu -text
.gitattributes:src/cuda/src/diagnostics.cuh -text
.gitattributes:src/cuda/src/diff.cu -text
.gitattributes:src/cuda/src/diff.cuh -text
.gitattributes:src/cuda/src/forcing.cu -text
.gitattributes:src/cuda/src/forcing.cuh -text
.gitattributes:src/cuda/src/gpu_astaroth.cu -text
.gitattributes:src/cuda/src/gpu_astaroth.cuh -text
.gitattributes:src/cuda/src/gpu_astaroth_v2.cu -text
.gitattributes:src/cuda/src/hydro.cu -text
.gitattributes:src/cuda/src/hydro.cuh -text
.gitattributes:src/cuda/src/init.conf -text
.gitattributes:src/cuda/src/initutils.cpp -text
.gitattributes:src/cuda/src/initutils.h -text
.gitattributes:src/cuda/src/integrators.cuh -text
.gitattributes:src/cuda/src/integrators_v5.cu -text
.gitattributes:src/cuda/src/integrators_v5s.cu -text
.gitattributes:src/cuda/src/io.cpp -text
.gitattributes:src/cuda/src/io.h -text
.gitattributes:src/cuda/src/libastaroth_sgl.so -text
.gitattributes:src/cuda/src/makefile -text
.gitattributes:src/cuda/src/makefile[!!-~](copy) -text
.gitattributes:src/cuda/src/makefile.depend -text
.gitattributes:src/cuda/src/makefile.local -text
.gitattributes:src/cuda/src/model_collectiveops.cpp -text
.gitattributes:src/cuda/src/model_collectiveops.h -text
.gitattributes:src/cuda/src/plotting_tools.py -text
.gitattributes:src/cuda/src/powerspectrum.py -text
.gitattributes:src/cuda/src/read_k_used.py -text
.gitattributes:src/cuda/src/read_k_used_view.py -text
.gitattributes:src/cuda/src/read_snapshot.py -text
.gitattributes:src/cuda/src/read_ts.py -text
.gitattributes:src/cuda/src/read_ts_show.py -text
.gitattributes:src/cuda/src/readme.txt -text
.gitattributes:src/cuda/src/refzplane0.txt -text
.gitattributes:src/cuda/src/run.conf -text
.gitattributes:src/cuda/src/shear.cu -text
.gitattributes:src/cuda/src/shear.cuh -text
.gitattributes:src/cuda/src/shear_old_BACKUP.cu -text
.gitattributes:src/cuda/src/slice.cu -text
.gitattributes:src/cuda/src/slice.cuh -text
.gitattributes:src/cuda/src/smem.cuh -text
.gitattributes:src/cuda/src/test.txt -text
.gitattributes:src/cuda/src/testmain.c -text
.gitattributes:src/cuda/src/timestep.cu -text
.gitattributes:src/cuda/src/timestep.cuh -text
.gitattributes:src/cuda/src/verification/model_collectiveops.cpp -text
.gitattributes:src/cuda/src/verification/model_collectiveops.h -text
.gitattributes:src/cuda/src/visual_data.py -text
.gitattributes:src/cuda/src/vtk_convert.py -text
.gitattributes:src/cuda/src/vuori_template.sh -text
.gitattributes:src/cuda/src/yt_visual.py -text
.gitattributes:src/cuda/src_new/CMakeLists.txt -text
.gitattributes:src/cuda/src_new/Makefile -text
.gitattributes:src/cuda/src_new/Makefile.depend -text
.gitattributes:src/cuda/src_new/README -text
.gitattributes:src/cuda/src_new/common/CMakeLists.txt -text
.gitattributes:src/cuda/src_new/common/config.cc -text
.gitattributes:src/cuda/src_new/common/config.h -text
.gitattributes:src/cuda/src_new/common/datatypes.h -text
.gitattributes:src/cuda/src_new/common/defines.h -text
.gitattributes:src/cuda/src_new/common/defines_PC.h -text
.gitattributes:src/cuda/src_new/common/defines_dims_PC.h -text
.gitattributes:src/cuda/src_new/common/errorhandler.h -text
.gitattributes:src/cuda/src_new/common/forcing.h -text
.gitattributes:src/cuda/src_new/common/grid.cc -text
.gitattributes:src/cuda/src_new/common/grid.h -text
.gitattributes:src/cuda/src_new/common/qualify.h -text
.gitattributes:src/cuda/src_new/common/slice.cc -text
.gitattributes:src/cuda/src_new/common/slice.h -text
.gitattributes:src/cuda/src_new/diagnostics/diagnostics.h -text
.gitattributes:src/cuda/src_new/diagnostics/timeseries_diagnostics.cc -text
.gitattributes:src/cuda/src_new/gpu/CMakeLists.txt -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/cuda_core.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/cuda_core.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/dconsts_core.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/cuda_generic.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/diff_cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cuh -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu -text
.gitattributes:src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cuh -text
.gitattributes:src/cuda/src_new/gpu/gpu.cc -text
.gitattributes:src/cuda/src_new/gpu/gpu.h -text
.gitattributes:src/cuda/src_new/gpu_astaroth.cc -text
.gitattributes:src/cuda/src_new/gpu_astaroth_new.cc -text
.gitattributes:src/cuda/src_new/howtoadd_makefiledepend_to_cmakelists.txt -text
.gitattributes:src/cuda/src_new/utils/utils.h -text
.gitattributes:src/gpu.h -text
.gitattributes:src/gpu.sed -text
.gitattributes:src/gpu_astaroth.cu -text
.gitattributes:src/gpu_astaroth.f90 -text
.gitattributes:src/gpu_astaroth_ansi.c -text
.gitattributes:src/nogpu.f90 -text
samples/gputest/job:#SBATCH -p gpu
samples/gputest/job:#SBATCH --gres=gpu:k80:1
samples/gputest/job:module load gcc/6.2.0 cmake/3.5.2 openmpi/2.1.2 cuda/9.0 
samples/gputest/job:#srun --cpus-per-task=1 -N 1 -n 1 --ntasks-per-node=1 --gres=gpu:k80:1 ./run.csh
samples/gputest/README:ln -s /cfs/klemming/home/b/brandenb/scr/GPU-test/pencil-code/src/astaroth/submodule/acc-runtime/samples/gputest submodule/acc-runtime/samples/gputest
samples/gputest/README:module load rocm/5.0.2 craype-accel-amd-gfx90a cmake/3.20.1
samples/gputest/README:git checkout gputestv5
samples/gputest/README:cd ../../../samples/gputest
samples/gputest/README:module load rocm craype-accel-amd-gfx90a cmake/3.20.1
samples/gputest/src/Makefile.local:GPU       =   gpu_astaroth
doc/figs/github-timetrace.eps:)_gUcP%?p>gPuD"5p:?Lor,;Sol>[SCuric[/kQ0[6h6ME*V6%qaQ^fl64$4\(Q:
doc/figs/github-timetrace.eps:MOR(?/p.m;No=9m"]`,@k5Ob-gpuP"@+bW6i%^an#:-*6ir&)VTegS5mHW2q./1s
doc/figs/cvsstat.eps:;S/6O8lCJ0V'q0m;M^R6.#CPoLm<t/fm.KZVd!?X?H5s/!`6"m:p4B:8MGPu5.-Q
doc/figs/cvsstat.eps:oVi.r%,F#Y5>d$3"l`UjJ:*HGpuV3'25]:1W#23S]cs%j5gZc[KB55/t+WhB++^F
doc/paper/paper.bib:    title = "{A new GPU-accelerated hydrodynamical code for numerical simulation of interacting galaxies}",
doc/paper/paper.bib:        title = "{Methods for compressible fluid simulation on GPUs using high-order
doc/citations/notes.tex:2023ApJ...959...32V,%V{\"a}is{\"a}l{\"a}}+ "Exploring the Formation of Resistive Pseudodisks with the GPU Code Astaroth"
doc/citations/notes.tex:2022ParC..11102904P,%Pekkila+ "Scalable communication for high-order stencil computations using CUDA-aware MPI"
doc/citations/notes.tex:2021ApJ...907...83V%Vaisala "Interaction of Large- and Small-scale Dynamos in Isotropic Turbulent Flows from GPU-accelerated Simulations" WO
doc/citations/notes.tex:2017CoPhC.217...11P,% Pekkila+ "Methods for compressible fluid simulation on GPUs using high-order finite differences"
doc/citations/notes.tex:{\em Code development, GPU etc} \citep{
doc/citations/notes.tex:2022ParC..11102904P,%Pekkila+ "Scalable communication for high-order stencil computations using CUDA-aware MPI"
doc/citations/notes.tex:2017CoPhC.217...11P% Pekkila+ "Methods for compressible fluid simulation on GPUs using high-order finite differences"
doc/citations/notes.tex:2023ApJ...959...32V,%V{\"a}is{\"a}l{\"a}}+ "Exploring the Formation of Resistive Pseudodisks with the GPU Code Astaroth"
doc/citations/notes.tex:2021ApJ...907...83V,%Vaisala "Interaction of Large- and Small-scale Dynamos in Isotropic Turbulent Flows from GPU-accelerated Simulations"
doc/citations/notes.tex:2013arXiv1311.0861I,% Kulikov, Igor "A new GPU-accelerated hydrodynamical code for numerical simulation of interacting galaxies" 
doc/citations/ref.bib:    title = "{A new GPU-accelerated hydrodynamical code for numerical simulation of interacting galaxies}",
doc/citations/ref.bib:        title = "{Methods for compressible fluid simulation on GPUs using high-order
doc/citations/ref.bib:        title = "{Interaction of Large- and Small-scale Dynamos in Isotropic Turbulent Flows from GPU-accelerated Simulations}",
doc/citations/ref.bib:     keywords = {Magnetic fields, Magnetohydrodynamics, Astrophysical fluid dynamics, Computational methods, GPU computing, 994, 1964, 101, 1965, 1969, Physics - Fluid Dynamics, Astrophysics - Solar and Stellar Astrophysics, Physics - Computational Physics},
doc/citations/ref.bib:        title = "{Scalable communication for high-order stencil computations using CUDA-aware MPI}",
doc/citations/ref.bib:        title = "{Exploring the Formation of Resistive Pseudodisks with the GPU Code Astaroth}",
doc/citations/ref.bib:        title = "{Scalable communication for high-order stencil computations using CUDA-aware MPI}",
doc/citations/ref.bib:        title = "{Exploring the Formation of Resistive Pseudodisks with the GPU Code Astaroth}",
doc/inlinedoc-modules.tex:%%   gpu_astaroth.f90
doc/inlinedoc-modules.tex:%%   nogpu.f90
doc/inlinedoc-modules.tex:  \var{gpu_astaroth.f90} & This module contains GPU related types and functions to be used with the ASTAROTH nucleus. \\
doc/inlinedoc-modules.tex:  \var{nogpu.f90} & This module contains GPU related dummy types and functions. \\
bin/pc_run:      $ENV{MPICH_GPU_SUPPORT_ENABLED} = '1';
bin/pc_run:    # GPU from Makefile.local
bin/pc_run:    $settings{'gpu'} = 0 + defined(match_line_ix(
bin/pc_run:        '^ \s* GPU *= *[a-zA-Z_]', 'src/Makefile.local'));
bin/pc_run:    identify_setting('gpu', %settings);
bin/pc_run:    if ($settings{'mpi'} && $settings{'gpu'}) {
bin/pc_setupsrc:make_cuda_dir()
bin/pc_setupsrc:## cuda subdirectory
bin/pc_setupsrc:  [ -h $src/cuda ] && rm $src/cuda
bin/pc_setupsrc:  adddir $src/cuda
bin/pc_setupsrc:## cuda/src_new subdirectory
bin/pc_setupsrc:  [ -h $src/cuda/src_new ] && rm $src/cuda/src_new
bin/pc_setupsrc:  adddir $src/cuda/src_new
bin/pc_setupsrc:  adddir $src/cuda/src_new/utils
bin/pc_setupsrc:  adddir $src/cuda/src_new/common
bin/pc_setupsrc:  adddir $src/cuda/src_new/gpu
bin/pc_setupsrc:  adddir $src/cuda/src_new/diagnostics
bin/pc_setupsrc:  adddir $src/cuda/src_new/gpu/cuda
bin/pc_setupsrc:  adddir $src/cuda/src_new/gpu/cuda/core
bin/pc_setupsrc:  adddir $src/cuda/src_new/gpu/cuda/generic
bin/pc_setupsrc:#  len=`expr match "$srcdir/cuda/astaroth/" '.*astaroth/'`
bin/pc_setupsrc:link_cuda_files()
bin/pc_setupsrc:#  LINK all *.* to local src/cuda directory
bin/pc_setupsrc:  echo "Linking files in '$src/cuda'."
bin/pc_setupsrc:  cd $src/cuda
bin/pc_setupsrc:  for file in $srcdir/cuda/src/*.cc $srcdir/cuda/src/*.cuh $srcdir/cuda/src/*.cpp $srcdir/cuda/src/*.h $srcdir/cuda/src/makefile $srcdir/cuda/src/makefile.* $srcdir/cuda/src/*.conf 
bin/pc_setupsrc:  echo "Linking files in '$src/cuda/src_new'."
bin/pc_setupsrc:  cd $src/cuda/src_new
bin/pc_setupsrc:  len=`expr match "$srcdir/cuda/src_new/" '.*src_new/'`
bin/pc_setupsrc:  for file in $srcdir/cuda/src_new/*.cc $srcdir/cuda/src_new/*.txt $srcdir/cuda/src_new/Makefile* $srcdir/cuda/src_new/utils/*.h $srcdir/cuda/src_new/common/*.cc $srcdir/cuda/src_new/common/*.txt $srcdir/cuda/src_new/common/*.h $srcdir/cuda/src_new/gpu/*.cc $srcdir/cuda/src_new/gpu/*.txt $srcdir/cuda/src_new/gpu/*.h $srcdir/cuda/src_new/diagnostics/*.h $srcdir/cuda/src_new/diagnostics/*.cc $srcdir/cuda/src_new/gpu/cuda/*.cc* $srcdir/cuda/src_new/gpu/cuda/core/*.cc* $srcdir/cuda/src_new/gpu/cuda/generic/*.cu*
bin/pc_setupsrc:  if [ "`grep '^[^#]*= *gpu' $src/Makefile.local`" != "" ]; then 
src/register.f90:      use GPU,              only: initialize_gpu
src/register.f90:      call initialize_gpu
src/register.f90:      use Gpu,            only: finalize_gpu
src/register.f90:      call finalize_gpu
src/general.f90:  public :: compress_nvidia
src/general.f90:    subroutine compress_nvidia(buffer,radius)
src/general.f90:    endsubroutine compress_nvidia
src/persist.f90:            if (.not.done.and.lgpu) dt=dtmp
src/Makefile.src:GPU                = nogpu
src/Makefile.src:LIBGPU =
src/Makefile.src:GPU_DEPS =
src/Makefile.src:    CUDA_LOADED := $(shell (module list |& grep -i cuda) )
src/Makefile.src:    ifeq ($(CUDA_LOADED),)
src/Makefile.src:      $(warning  No CUDA module seems to be loaded (needed for training)!)
src/Makefile.src:      $(info Try (one of) the following: $(shell (module spider |& grep -i cuda | sed -e's/^.* \([^ ]*[Cc][Uu][Dd][Aa][^ ]*\) .*$$/\1/') ) )
src/Makefile.src:    override FFLAGS+=-cuda -noswitcherror -I$(TORCHFORT_PATH)/include/
src/Makefile.src:    override LDFLAGS+=-L$(TORCHFORT_PATH)/lib/ -ltorchfort_fort -ltorchfort -cuda
src/Makefile.src:ifneq ($(GPU),nogpu)
src/Makefile.src:    CUDA_LOADED := $(shell (module list |& grep -i cuda) )
src/Makefile.src:    ROCM_LOADED := $(shell (module list |& grep -i rocm) )
src/Makefile.src:    ifeq ($(CUDA_LOADED)$(ROCM_LOADED),)
src/Makefile.src:      $(warning  No CUDA or ROCM module seems to be loaded!)
src/Makefile.src:      $(info Try (one of) the following: $(shell (module spider |& grep -i cuda | sed -e's/^.* \([^ ]*[Cc][Uu][Dd][Aa][^ ]*\) .*$$/\1/') ) )
src/Makefile.src:      $(info or: $(shell (module spider |& grep -i rocm | sed -e's/^.* \([^ ]*[Rr][Oo][Cc][Mm][^ ]*\) .*$$/\1/') ) )
src/Makefile.src:  ifneq ($(findstring gpu_astaroth, $(GPU)),)
src/Makefile.src:      GPU_INTERFACE = gpu_astaroth_ansi.o
src/Makefile.src:        LIBGPU = astaroth_dbl.so
src/Makefile.src:        LIBGPU = astaroth_sgl.so
src/Makefile.src:#  GPU_DEPS = $(LIBGPU) gpu_preps
src/Makefile.src:      GPU_DEPS = gpu_preps
src/Makefile.src:      LDFLAGS_LIB = -L astaroth -L astaroth/submodule/build/src/core -L astaroth/submodule/build/src/core/kernels -L astaroth/submodule/build/src/utils -lastaroth_core -lkernels -lastaroth_utils -l:$(LIBGPU)
src/Makefile.src:    $(error No other GPU interfaces than gpu_astaroth* recently supported!)
src/Makefile.src:  GPU_INTERFACE=
src/Makefile.src:GPU_OBJ=$(GPU).o
src/Makefile.src:GPU_SRC=$(GPU).f90
src/Makefile.src:## Taito-gpu: with GPUs - (Kajaani)
src/Makefile.src:#FC=mpif90 #(taito-gpu)
src/Makefile.src:#FSTD_95=-std95 #(taito-gpu)
src/Makefile.src:#F90=$(FC) #(taito-gpu)
src/Makefile.src:#CC=mpicc #(taito-gpu)
src/Makefile.src:#FFLAGS=-O1 -fbacktrace #(taito-gpu)
src/Makefile.src:#FFLAGS_DOUBLE=-fdefault-real-8 #(taito-gpu)
src/Makefile.src:#CFLAGS=-DFUNDERSC=1 -O3 #(taito-gpu)
src/Makefile.src:preprocess=$(ASCALAR_SRC) $(BORDER_PROFILES_SRC) $(CHEMISTRY_SRC) $(CHIRAL_SRC) $(COSMICRAY_SRC) $(COSMICRAYFLUX_SRC) $(DENSITY_SRC) $(DERIV_SRC) $(DETONATE_SRC) $(DUSTDENSITY_SRC) $(DUSTVELOCITY_SRC) $(ENERGY_SRC) $(EOS_SRC) $(FIXED_POINT_SRC) $(FORCING_SRC) $(GRAVITY_SRC) $(GPU_SRC) $(GRID_SRC) $(HEATFLUX_SRC) $(HYDRO_SRC) $(HYPERRESI_STRICT_SRC) $(HYPERVISC_STRICT_SRC) $(IMPLICIT_DIFFUSION_SRC) $(IMPLICIT_PHYSICS_SRC) $(INITIAL_CONDITION_SRC) $(INTERSTELLAR_SRC) $(LORENZ_GAUGE_SRC) $(MAGNETIC_SRC) $(MAGNETIC_MEANFIELD_SRC) $(MAGNETIC_MEANFIELD_DEMFDT_SRC) $(MPICOMM_SRC) $(NEUTRALDENSITY_SRC) $(NEUTRALVELOCITY_SRC) $(NSCBC_SRC) $(OPACITY_SRC) $(PARTICLES_SRC) $(PARTICLES_ADAPTATION_SRC) $(PARTICLES_COAGULATION_SRC) $(PARTICLES_CONDENSATION_SRC)  $(PARTICLES_COLLISIONS_SRC) $(PARTICLES_MAP_SRC) $(PARTICLES_DENSITY_SRC)  $(PARTICLES_MASS_SRC) $(PARTICLES_NUMBER_SRC) $(PARTICLES_RADIUS_SRC) $(PARTICLES_POTENTIAL_SRC) $(PARTICLES_GRAD_SRC) $(PARTICLES_SELFGRAVITY_SRC) $(PARTICLES_SINK_SRC) $(PARTICLES_DRAG_SRC) $(PARTICLES_SPIN_SRC) $(PARTICLES_STALKER_SRC) $(PARTICLES_LYAPUNOV_SRC) $(PARTICLES_CAUSTICS_SRC) $(PARTICLES_TETRAD_SRC) $(PARTICLES_STIRRING_SRC) $(PARTICLES_DIAGNOS_DV_SRC) $(PARTICLES_DIAGNOS_STATE_SRC) $(PARTICLES_PERSISTENCE_SRC) $(PARTICLES_TEMPERATURE_SRC) $(PARTICLES_ADSORBED_SRC) $(PARTICLES_SURFSPEC_SRC) $(PARTICLES_CHEMISTRY_SRC) $(POINTMASSES_SRC) $(POISSON_SRC) $(POLYMER_SRC) $(POWER_SRC) $(PYTHON_SRC) $(PSCALAR_SRC) $(RADIATION_SRC) $(SELFGRAVITY_SRC) $(SGSHYDRO_SRC) $(SHEAR_SRC) $(SHOCK_SRC) $(SIGNAL_HANDLING_SRC) $(SOLID_CELLS_SRC) $(STREAMLINES_SRC) $(TESTFIELD_GENERAL_SRC) $(TESTFIELD_SRC) $(TESTFLOW_SRC) $(TESTSCALAR_SRC) $(TIMEAVG_SRC) $(TRAINING_SRC) $(VISCOSITY_SRC) cparam.local
src/Makefile.src:start=$(technical) $(physics) $(PARAM_IO_OBJ) nogpu.o
src/Makefile.src:run=$(technical) $(physics) $(GPU_OBJ) $(GPU_INTERFACE) $(TIMESTEP_OBJ) equ.o pencil_check.o $(PARAM_IO_OBJ)
src/Makefile.src:run.x: run.o | $(SPECIAL_DEPS) $(GPU_DEPS)
src/Makefile.src:#needs LIBGPU as prerequisite, this in turn needs a rule
src/Makefile.src:#$(LIBGPU):
src/Makefile.src:# writes moduleflags in $(CUDA_INCDIR)/PC_moduleflags.h, list of needed physics module source files in $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:CUDA_MAKEDIR=astaroth
src/Makefile.src:CUDA_INCDIR=$(CUDA_MAKEDIR)
src/Makefile.src:CUDA_RELPATH=..
src/Makefile.src:DSL_WORKDIR=$(CUDA_MAKEDIR)/submodule/acc-runtime/Pencil
src/Makefile.src:gpu_preps: $(CUDA_INCDIR)/PC_moduleflags.h $(CUDA_MAKEDIR)/PC_modulesources.h $(DSL_WORKDIR)/fieldecs.h $(DSL_WORKDIR)/equations.h $(DSL_WORKDIR)/solve.ac
src/Makefile.src:#$(CUDA_INCDIR)/PC_moduleflags.h $(CUDA_MAKEDIR)/PC_modulesources.h $(DSL_WORKDIR)/fieldecs.h $(DSL_WORKDIR)/equations.h $(DSL_WORKDIR)/solve.ac:: .sentinel
src/Makefile.src:$(CUDA_INCDIR)/PC_moduleflags.h $(CUDA_MAKEDIR)/PC_modulesources.h $(DSL_WORKDIR)/fieldecs.h $(DSL_WORKDIR)/equations.h $(DSL_WORKDIR)/solve.ac: .sentinel
src/Makefile.src:	@rm -f $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile.src:	@echo '#pragma once' > $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile.src:	@sed -e's/CUDA_RELPATH/'$(CUDA_RELPATH)'/' -e's/CUDA_MAKEDIR/'$(CUDA_MAKEDIR)'/' < gpu.sed > gpu.sed.tmp
src/Makefile.src:	@sed -f gpu.sed.tmp < Makefile.local >> $(CUDA_INCDIR)/PC_moduleflags.h; rm -f gpu.sed.tmp
src/Makefile.src:	@echo '#define LVISCOSITY 1 // '$(CUDA_RELPATH)'/viscosity.f90' >> $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile.src:	@echo $(CUDA_RELPATH)'/viscosity.f90 \' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:	@echo $(CUDA_RELPATH)'/timestep.f90 \' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:	@echo $(CUDA_RELPATH)'/noeos.f90' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:	@echo ' ' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:	@sed -e's/MODULESOURCES=/MODULESOURCES="/' -e'$$ s/$$/"/' -e's/\.\.\///' $(CUDA_MAKEDIR)/PC_modulesources.h > tmp; source ./tmp; rm tmp; \
src/Makefile.src:#	@sed -e':loop' -e'N' -e'$$!b loop' -e's/\#pragma  *once//' -e's/\# *define *\([A-Z0-9_][A-Z0-9_]*\)/..\/..\/\L\1.f90 /g' -e's/\n/ /g' < $(CUDA_INCDIR)/PC_moduleflags.h >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile.src:technical_nompi=$(CPARAM_OBJ) $(CDATA_OBJ) $(NOMPICOMM_OBJ) $(BOUNDCOND_OBJ) $(BORDER_PROFILES_OBJ) $(DEBUG_OBJ) $(DERIV_OBJ) $(FOURIER_OBJ) $(GEOMETRICAL_TYPES_OBJ) $(GHOSTFOLD_OBJ) nogpu.o $(GRID_OBJ) $(GSL_OBJ) $(INITIAL_CONDITION_OBJ) $(DEBUG_IO_OBJ) $(HDF5_IO_OBJ) $(IO_PP_OBJ) $(FILE_IO_OBJ) $(FIXED_POINT_OBJ) $(LSODE_FC_OBJ) $(NSCBC_OBJ) $(POWER_OBJ) $(PYTHON_OBJ) $(SIGNAL_HANDLING_OBJ) $(SLICES_OBJ) $(SLICES_METHODS_OBJ) $(solid_cells_objects) $(STRUCT_FUNC_OBJ) $(SYSCALLS_OBJ) syscalls_ansi.o $(TESTPERTURB_OBJ) $(TIMEAVG_OBJ) $(WENO_TRANSPORT_OBJ) $(DIAGNOSTICS_OBJ) $(FARRAY_OBJ) $(FILTER_OBJ) $(INITCOND_OBJ) $(MESSAGES_OBJ) $(GENERAL_OBJ) $(PERSIST_OBJ) $(REGISTER_OBJ) $(SHARED_VARIABLES_OBJ) $(SNAPSHOT_OBJ) $(STREAMLINES_OBJ) $(SUB_OBJ) noyinyang.o noyinyang_mpi.o $(GHOST_CHECK_OBJ) $(IMPLICIT_DIFFUSION_OBJ) $(TRAINING_OBJ) magnetic.a $(PARTICLES_MAIN).a
src/Makefile.src:ifneq ($(and $(IS_CRAY), $(filter-out $(GPU),nogpu)),)
src/Makefile.src:ifneq ($(and $(IS_CRAY), $(filter-out $(GPU),nogpu)),)
src/Makefile.src:	rm -f .sentinel $(CUDA_MAKEDIR)/PC_modulesources.h $(CUDA_INCDIR)/PC_moduleflags.h  # for GPU
src/initial_condition/disk+corona.f90:!          procno=iprocmask2 = abs(XX) gt 4.0
src/gpu_astaroth.f90:! MODULE_DOC: This module contains GPU related types and functions to be used with the ASTAROTH nucleus.
src/gpu_astaroth.f90:! CPARAM logical, parameter :: lgpu = .true.
src/gpu_astaroth.f90:module GPU
src/gpu_astaroth.f90:  external initialize_gpu_c
src/gpu_astaroth.f90:  external finalize_gpu_c
src/gpu_astaroth.f90:  external rhs_gpu_c
src/gpu_astaroth.f90:  include 'gpu.h'
src/gpu_astaroth.f90:  !integer(KIND=ikind8) :: pFarr_GPU_in, pFarr_GPU_out
src/gpu_astaroth.f90:  type(C_PTR) :: pFarr_GPU_in, pFarr_GPU_out
src/gpu_astaroth.f90:    subroutine initialize_GPU
src/gpu_astaroth.f90:      if (str/='') call stop_it('No GPU implementation for module(s) "'//trim(str(3:))//'"')
src/gpu_astaroth.f90:      call initialize_gpu_c(pFarr_GPU_in,pFarr_GPU_out)
src/gpu_astaroth.f90:!print'(a,1x,Z0,1x,Z0)', 'pFarr_GPU_in,pFarr_GPU_out=', pFarr_GPU_in,pFarr_GPU_out
src/gpu_astaroth.f90:    endsubroutine initialize_GPU
src/gpu_astaroth.f90:    subroutine gpu_init
src/gpu_astaroth.f90:      call init_gpu_c
src/gpu_astaroth.f90:    endsubroutine gpu_init
src/gpu_astaroth.f90:    subroutine register_GPU(f)
src/gpu_astaroth.f90:      call register_gpu_c(f)
src/gpu_astaroth.f90:    endsubroutine register_GPU
src/gpu_astaroth.f90:    subroutine finalize_GPU
src/gpu_astaroth.f90:      call finalize_gpu_c
src/gpu_astaroth.f90:    endsubroutine finalize_GPU
src/gpu_astaroth.f90:    subroutine rhs_GPU(f,isubstep,early_finalize)
src/gpu_astaroth.f90:      call rhs_gpu_c(isubstep,lvery_first,early_finalize)
src/gpu_astaroth.f90:    endsubroutine rhs_GPU
src/gpu_astaroth.f90:    function get_ptr_GPU(ind1,ind2,lout) result(pFarr)
src/gpu_astaroth.f90:        call c_f_pointer(pos_real_ptr_c(pFarr_GPU_out,ind1-1),pFarr,(/mx,my,mz,i2-ind1+1/))
src/gpu_astaroth.f90:        call c_f_pointer(pos_real_ptr_c(pFarr_GPU_in,ind1-1),pFarr,(/mx,my,mz,i2-ind1+1/))
src/gpu_astaroth.f90:    endfunction get_ptr_GPU
src/gpu_astaroth.f90:    subroutine copy_farray_from_GPU(f)
src/gpu_astaroth.f90:    endsubroutine copy_farray_from_GPU
src/gpu_astaroth.f90:endmodule GPU
src/interstellar.f90:                    if (.not.lgpu.and.ip==1963) &
src/farray_alloc.f90:    if (nt>0.and..not.lgpu) then
src/Makefile.depend:$(REGISTER_OBJ): $(OMP_PREREQ) $(REGISTER_SRC) $(CDATA_OBJ) $(FARRAY_OBJ) $(MPICOMM_OBJ) $(SUB_OBJ) $(IO_OBJ) $(physics) $(TIMEAVG_OBJ) $(DIAGNOSTICS_OBJ) $(EQU_OBJ) $(GPU_OBJ) $(GRID_OBJ) $(TESTPERTURB_OBJ) $(SOLID_CELLS_OBJ) $(DIAGNOSTICS_OBJ) $(PARAM_IO_OBJ) $(POWER_OBJ) $(TRAINING_OBJ)
src/Makefile.depend:$(EQU_OBJ): $(EQU_SRC) $(CDATA_OBJ) $(MPICOMM_OBJ) $(GHOSTFOLD_OBJ) $(GPU_OBJ) $(MESSAGES_OBJ) $(SUB_OBJ) $(physics) $(BOUNDCOND_OBJ) $(POISSON_OBJ) $(PARTICLES_MAIN_OBJ) $(INTERSTELLAR_OBJ) $(GRID_OBJ) $(SNAPSHOT_OBJ) $(FORCING_OBJ) $(SOLID_CELLS_OBJ) $(NSCBC_OBJ) $(DIAGNOSTICS_OBJ) $(HEATFLUX_OBJ) $(TRAINING_OBJ)
src/Makefile.depend:$(SNAPSHOT_OBJ): $(SNAPSHOT_SRC) $(BOUNDCOND_OBJ) $(CDATA_OBJ) $(EOS_OBJ) $(MESSAGES_OBJ) $(MPICOMM_OBJ) $(GPU_OBJ) $(IO_OBJ) $(PARTICLES_MAIN_OBJ) $(PERSIST_OBJ) $(POWER_OBJ) $(PSCALAR_OBJ) $(RADIATION_OBJ) $(SHOCK_OBJ) $(STRUCT_FUNC_OBJ)
src/Makefile.depend:$(TRAINING_OBJ): training.h $(TRAINING_SRC) $(CDATA_OBJ) $(DIAGNOSTICS_OBJ) $(FILE_IO_OBJ) $(FARRAY_OBJ) $(GENERAL_OBJ) $(GPU_OBJ) $(MESSAGES_OBJ) $(MPICOMM_OBJ) $(SYSCALLS_OBJ)
src/Makefile.depend:$(GPU_OBJ): $(GPU_INTERFACE) gpu.h $(GPU_SRC) $(CPARAM_OBJ) $(CDATA_OBJ) $(GENERAL_OBJ) $(MPICOMM_OBJ) Makefile.src
src/Makefile.depend:nogpu.o: gpu.h nogpu.f90 $(GENERAL_OBJ)
src/Makefile.depend:gpu_astaroth_ansi.o: gpu_astaroth_ansi.c headers_c.h Makefile.local
src/Makefile:GPU                = nogpu
src/Makefile:LIBGPU =
src/Makefile:GPU_DEPS =
src/Makefile:ifeq ($(GPU),gpu_astaroth)
src/Makefile:    CUDA_LOADED := $(shell (module list |& grep -i cuda) )
src/Makefile:    ifeq ($(CUDA_LOADED),)
src/Makefile:no_cuda_message:
src/Makefile:	@echo ";;; No CUDA module seems to be loaded! Exit."
src/Makefile:  GPU_INTERFACE = gpu_astaroth_ansi.o
src/Makefile:  GPU_DEPS = gpu_preps astaroth/submodule/build/src/core/libastaroth_core.so
src/Makefile:    LIBGPU = astaroth_dbl.so
src/Makefile:    LIBGPU = astaroth_sgl.so
src/Makefile:  LDFLAGS_LIB = -L astaroth -L astaroth/submodule/build/src/core -lastaroth_core -l:$(LIBGPU)
src/Makefile:  GPU_INTERFACE=
src/Makefile:GPU_OBJ=$(GPU).o
src/Makefile:GPU_SRC=$(GPU).f90
src/Makefile:## Taito-gpu: with GPUs - (Kajaani)
src/Makefile:#FC=mpif90 #(taito-gpu)
src/Makefile:#FSTD_95=-std95 #(taito-gpu)
src/Makefile:#F90=$(FC) #(taito-gpu)
src/Makefile:#CC=mpicc #(taito-gpu)
src/Makefile:#FFLAGS=-O1 -fbacktrace #(taito-gpu)
src/Makefile:#FFLAGS_DOUBLE=-fdefault-real-8 #(taito-gpu)
src/Makefile:#CFLAGS=-DFUNDERSC=1 -O3 #(taito-gpu)
src/Makefile:preprocess=$(BORDER_PROFILES_SRC) $(CHEMISTRY_SRC) $(CHIRAL_SRC) $(COSMICRAY_SRC) $(COSMICRAYFLUX_SRC) $(DENSITY_SRC) $(DERIV_SRC) $(DETONATE_SRC) $(DUSTDENSITY_SRC) $(DUSTVELOCITY_SRC) $(ENERGY_SRC) $(EOS_SRC) $(FIXED_POINT_SRC) $(FORCING_SRC) $(GRAVITY_SRC) $(GPU_SRC) $(GRID_SRC) $(HEATFLUX_SRC) $(HYDRO_SRC) $(HYPERRESI_STRICT_SRC) $(HYPERVISC_STRICT_SRC) $(IMPLICIT_DIFFUSION_SRC) $(IMPLICIT_PHYSICS_SRC) $(INITIAL_CONDITION_SRC) $(INTERSTELLAR_SRC) $(LORENZ_GAUGE_SRC) $(MAGNETIC_SRC) $(MAGNETIC_MEANFIELD_SRC) $(MAGNETIC_MEANFIELD_DEMFDT_SRC) $(NEUTRALDENSITY_SRC) $(NEUTRALVELOCITY_SRC) $(NSCBC_SRC) $(OPACITY_SRC) $(PARTICLES_SRC) $(PARTICLES_ADAPTATION_SRC) $(PARTICLES_COAGULATION_SRC) $(PARTICLES_CONDENSATION_SRC)  $(PARTICLES_COLLISIONS_SRC) $(PARTICLES_MAP_SRC) $(PARTICLES_DENSITY_SRC)  $(PARTICLES_MASS_SRC) $(PARTICLES_NUMBER_SRC) $(PARTICLES_RADIUS_SRC) $(PARTICLES_POTENTIAL_SRC) $(PARTICLES_GRAD_SRC) $(PARTICLES_SELFGRAVITY_SRC) $(PARTICLES_SINK_SRC) $(PARTICLES_DRAG_SRC) $(PARTICLES_SPIN_SRC) $(PARTICLES_STALKER_SRC) $(PARTICLES_LYAPUNOV_SRC) $(PARTICLES_CAUSTICS_SRC) $(PARTICLES_TETRAD_SRC) $(PARTICLES_STIRRING_SRC) $(PARTICLES_DIAGNOS_DV_SRC) $(PARTICLES_DIAGNOS_STATE_SRC) $(PARTICLES_PERSISTENCE_SRC) $(PARTICLES_TEMPERATURE_SRC) $(PARTICLES_ADSORBED_SRC) $(PARTICLES_SURFSPEC_SRC) $(PARTICLES_CHEMISTRY_SRC) $(POINTMASSES_SRC) $(POISSON_SRC) $(POLYMER_SRC) $(POWER_SRC) $(PSCALAR_SRC) $(RADIATION_SRC) $(SELFGRAVITY_SRC) $(SHEAR_SRC) $(SHOCK_SRC) $(SIGNAL_HANDLING_SRC) $(SOLID_CELLS_SRC) $(SPECIAL_SRC) $(ASCALAR_SRC) $(STREAMLINES_SRC) $(TESTFIELD_GENERAL_SRC) $(TESTFIELD_SRC) $(TESTFLOW_SRC) $(TESTSCALAR_SRC) $(VISCOSITY_SRC) cparam.local
src/Makefile:start=$(physics) $(technical) nogpu.o
src/Makefile:run=$(physics) $(technical) $(GPU_OBJ) $(GPU_INTERFACE) $(TIMESTEP_OBJ) equ.o pencil_check.o
src/Makefile:run.x: run.o | $(SPECIAL_DEPS) $(GPU_DEPS)
src/Makefile:# writes moduleflags in $(CUDA_INCDIR)/PC_moduleflags.h, list of needed physics module source files in $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:CUDA_MAKEDIR=astaroth
src/Makefile:CUDA_INCDIR=$(CUDA_MAKEDIR)
src/Makefile:CUDA_RELPATH=..
src/Makefile:gpu_preps: $(CUDA_INCDIR)/PC_moduleflags.h $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:$(CUDA_INCDIR)/PC_moduleflags.h $(CUDA_MAKEDIR)/PC_modulesources.h: .sentinel
src/Makefile:	@rm -f $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile:	@echo '#pragma once' > $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile:	@sed -i -e's/CUDA_RELPATH/'$(CUDA_RELPATH)'/' -e's/CUDA_MAKEDIR/'$(CUDA_MAKEDIR)'/' $(PENCIL_HOME)/src/gpu.sed
src/Makefile:	@sed -f gpu.sed < Makefile.local >> $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile:	@echo '#define LVISCOSITY 1 // '$(CUDA_RELPATH)'/viscosity.f90' >> $(CUDA_INCDIR)/PC_moduleflags.h
src/Makefile:	@echo $(CUDA_RELPATH)'/viscosity.f90 \' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:	@echo $(CUDA_RELPATH)'/timestep.f90 \' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:	@echo $(CUDA_RELPATH)'/noeos.f90' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:	@echo '' >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:#	@sed -e':loop' -e'N' -e'$$!b loop' -e's/\#pragma  *once//' -e's/\# *define *\([A-Z0-9_][A-Z0-9_]*\)/..\/..\/\L\1.f90 /g' -e's/\n/ /g' < $(CUDA_INCDIR)/PC_moduleflags.h >> $(CUDA_MAKEDIR)/PC_modulesources.h
src/Makefile:technical_nompi=$(BOUNDCOND_OBJ) $(BORDER_PROFILES_OBJ) $(DEBUG_OBJ) $(DERIV_OBJ) $(FOURIER_OBJ) $(GEOMETRICAL_TYPES_OBJ) $(GHOSTFOLD_OBJ) nogpu.o $(GRID_OBJ) $(GSL_OBJ) $(INITIAL_CONDITION_OBJ) $(DEBUG_IO_OBJ) $(HDF5_IO_OBJ) $(IO_PP_OBJ) $(FILE_IO_OBJ) $(FIXED_POINT_OBJ) $(LSODE_FC_OBJ) $(NOMPICOMM_OBJ) $(NSCBC_OBJ) $(POWER_OBJ) $(SIGNAL_HANDLING_OBJ) $(SLICES_OBJ) $(SLICES_METHODS_OBJ) $(solid_cells_objects) $(STRUCT_FUNC_OBJ) $(SYSCALLS_OBJ) syscalls_ansi.o $(TESTPERTURB_OBJ) $(TIMEAVG_OBJ) $(WENO_TRANSPORT_OBJ) $(CDATA_OBJ) $(CPARAM_OBJ) $(DIAGNOSTICS_OBJ) $(FARRAY_OBJ) $(FILTER_OBJ) $(INITCOND_OBJ) $(MESSAGES_OBJ) $(GENERAL_OBJ) $(PARAM_IO_OBJ) $(PERSIST_OBJ) $(REGISTER_OBJ) $(SHARED_VARIABLES_OBJ) $(SNAPSHOT_OBJ) $(STREAMLINES_OBJ) $(SUB_OBJ) noyinyang.o noyinyang_mpi.o $(GHOST_CHECK_OBJ) $(IMPLICIT_DIFFUSION_OBJ) magnetic.a $(PARTICLES_MAIN).a
src/Makefile:	$(LD) $(start) start.o $(LDFLAGS) $(LD_MPI) $(LD_FOURIER) $(LIBGPU) -o start.x
src/Makefile:	$(LD) $(run) run.o $(LDFLAGS) $(LDFLAGS_MAIN) $(LD_MPI) $(LD_FOURIER) $(LIBSIG) $(LIBGPU) -o run.x
src/Makefile:	rm -f .sentinel $(CUDA_MAKEDIR)/PC_modulesources.h $(CUDA_INCDIR)/PC_moduleflags.h  # for GPU
src/hydro.f90:        if (.not.lgpu) then
src/scripts/diagnostics2c:sed -e'/GPU-START/,/GPU-END/ !d' -e'/GPU-END/ d' \
src/scripts/diagnostics2c:    -e's/^.*GPU-START.*$/#include "headers_c.h"/' < diagnostics.f90 > diagnostics_c.h
src/scripts/diagnostics2c:# -e's/^ *! *GPU-START; *$/\n{/' -e's/^ *! *GPU-END; *$/}/'
src/scripts/phys_modules2c:#            s/call  *copy_addr *( *\([a-zA-Z0-9_]*\) *, *p_par(\([0-9]*\))) *$/CUDA_ERRCHK( cudaMemcpyToSymbol(d_\U\1\L, p_par_'$modname'[\2-1], sizeof(real)) );  \/\/ \1 real/w tmp1
src/scripts/phys_modules2c:#            s/call  *copy_addr *( *\([a-zA-Z0-9_]*\) *, *p_par(\([0-9]*\))) *! *\([a-zA-Z][a-zA-Z]*\) *$/CUDA_ERRCHK( cudaMemcpyToSymbol(d_\U\1\L, p_par_'$modname'[\2-1], sizeof(\3)) ); \/\/ \1 \3/w tmp1
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]\)\([mr][am][sx]\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2>0) {\n  diag=reduce_cuda_PC(\U\2_VEC\L, h_grid.\U\1\1X\L);\n  save_name(diag,idiag_\1\2);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]\)\(min\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2>0) {\n  diag=-reduce_cuda_PC(\U\2_VEC\L, h_grid.\U\1\1X\L);\n  save_name(diag,idiag_\1\2);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]\)\([xyz]\)\([mr][iam][nsx]\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2\3>0) {\n  diag=reduce_cuda_PC(\U\3_SCAL\L, h_grid.\U\1\1\2\L);\n  save_name(diag,idiag_\1\2\3);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]\)\([xyz]\)\(min\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2\3>0) {\n  diag=-reduce_cuda_PC(\U\3_SCAL\L, h_grid.\U\1\1\2\L);\n  save_name(diag,diag_\1\2\3);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]*\)\([mr][am][sx]\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2>0) {\n  diag=reduce_cuda_PC(\U\2_SCAL\L, h_grid.\U\1\L);\n  save_name(diag,idiag_\1\2);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]*\)\(min\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2>0) {\n  diag=-reduce_cuda_PC(\U\2_SCAL\L, h_grid.\U\1\L);\n  save_name(diag,idiag_\1\2);\n}/ 
src/scripts/phys_modules2c:             s/call  *copy_addr *( *idiag_\([a-zA-Z0-9_]*\)\(m\) *, *p_'$1' *( *\([0-9]*\) *) *) *$/if (idiag_\1\2>0) {\n  diag=reduce_cuda_PC(\U SUM_SCAL\L, h_grid.\U\1\L);\n  save_name(diag,idiag_\1\2);\n}/ 
src/timestep_rkf_lowsto.f90:!                  the GPU coupled code.
src/timestep_rkf_lowsto.f90:        if (.not. lgpu) then
src/gpu_astaroth_ansi.c:/*                             gpu_astaroth_ansi.c
src/gpu_astaroth_ansi.c:void initGPU();
src/gpu_astaroth_ansi.c:void registerGPU(REAL*);
src/gpu_astaroth_ansi.c:void initializeGPU();
src/gpu_astaroth_ansi.c:void finalizeGPU();
src/gpu_astaroth_ansi.c:void substepGPU(int isubstep, int full, int early_finalize);
src/gpu_astaroth_ansi.c:void FTNIZE(initialize_gpu_c)(REAL **farr_GPU_in, REAL **farr_GPU_out)
src/gpu_astaroth_ansi.c:/* Initializes GPU.
src/gpu_astaroth_ansi.c:  initializeGPU(farr_GPU_in,farr_GPU_out);
src/gpu_astaroth_ansi.c:void FTNIZE(init_gpu_c)()
src/gpu_astaroth_ansi.c:// Initializes GPU use.
src/gpu_astaroth_ansi.c:  initGPU();
src/gpu_astaroth_ansi.c:void FTNIZE(register_gpu_c)(REAL* f)
src/gpu_astaroth_ansi.c:// Allocates memory on GPU according to setup needs.
src/gpu_astaroth_ansi.c:  registerGPU(f);
src/gpu_astaroth_ansi.c:void FTNIZE(finalize_gpu_c)()
src/gpu_astaroth_ansi.c:// Frees memory allocated on GPU.
src/gpu_astaroth_ansi.c:  finalizeGPU();
src/gpu_astaroth_ansi.c:void FTNIZE(rhs_gpu_c)
src/gpu_astaroth_ansi.c:/* Communication between CPU and GPU: copy (outer) halos from CPU to GPU, 
src/gpu_astaroth_ansi.c:   copy "inner halos" from GPU to CPU; calculation of rhss of momentum eq.
src/gpu_astaroth_ansi.c:   and of continuity eq. by GPU kernels. Perform the Runge-Kutta substep 
src/gpu_astaroth_ansi.c:  substepGPU(*isubstep, *full, *early_finalize);
src/timestep.f90:          if (.not.lgpu) df=0.0
src/timestep.f90:          if (.not.lgpu) df=alpha_ts(itsub)*df !(could be subsumed into pde, but is dangerous!)
src/timestep.f90:!  With GPUs this is done on the CUDA side.
src/timestep.f90:        if (lfirst.and.ldt.and..not.lgpu) call set_dt(maxval(dt1_max))
src/timestep.f90:        if (.not. lgpu) f(l1:l2,m1:m2,n1:n2,1:mvar) =  f(l1:l2,m1:m2,n1:n2,1:mvar) &
src/cparam.local:integer, parameter :: ncpus=1,nprocx=1,nprocy=1,nprocz=ncpus/(nprocx*nprocy),ngpus=1
src/density.f90:        if (.not.lgpu) then
src/diagnostics_outlog.f90:! GPU-START
src/diagnostics_outlog.f90:! GPU-END
src/training_torchfort.f90:    use Cudafor
src/training_torchfort.f90:      istat = cudaSetDevice(iproc)
src/training_torchfort.f90:      if (istat /= CUDASUCCESS) call fatal_error('initialize_training','cudaSetDevice failed')
src/training_torchfort.f90:      if (.not.lgpu) then
src/training_torchfort.f90:      use Gpu, only: get_ptr_gpu
src/training_torchfort.f90:      if (.not.lgpu) then
src/training_torchfort.f90:        !call get_ptr_gpu(ptr_uu,iux,iuz)
src/training_torchfort.f90:        !call get_ptr_gpu(ptr_tau,tauxx,tauzz)
src/training_torchfort.f90:        istat = torchfort_inference(model, get_ptr_gpu(iux,iuz), get_ptr_gpu(itauxx,itauzz))
src/training_torchfort.f90:      elseif (.not.lgpu) then
src/training_torchfort.f90:      use Gpu, only: get_ptr_gpu
src/training_torchfort.f90:        if (lgpu) then
src/training_torchfort.f90:          istat = torchfort_inference(model, get_ptr_gpu(iux,iuz), get_ptr_gpu(itauxx,itauzz))
src/training_torchfort.f90:      if (.not.lgpu) deallocate(input,label,output)
src/snapshot.f90:  use Gpu, only: copy_farray_from_GPU
src/snapshot.f90:        if (.not.lstart.and.lgpu) call copy_farray_from_GPU(a)
src/snapshot.f90:          if (.not.lstart.and.lgpu) call copy_farray_from_GPU(a)
src/snapshot.f90:        if (.not.lstart.and.lgpu) call copy_farray_from_GPU(a)
src/snapshot.f90:        if (.not.lstart.and.lgpu) call copy_farray_from_GPU(f)
src/run.f90:  use Gpu,             only: gpu_init, register_gpu
src/run.f90:!  Initialize GPU use.
src/run.f90:  call gpu_init
src/run.f90:  call register_gpu(f)
src/run.f90:    if (lforcing.and..not.lgpu) call addforce(f)
src/nogpu.f90:! MODULE_DOC: This module contains GPU related dummy types and functions.
src/nogpu.f90:! CPARAM logical, parameter :: lgpu = .false.
src/nogpu.f90:module GPU
src/nogpu.f90:  include 'gpu.h'
src/nogpu.f90:    subroutine initialize_GPU
src/nogpu.f90:    endsubroutine initialize_GPU
src/nogpu.f90:    subroutine gpu_init
src/nogpu.f90:    endsubroutine gpu_init
src/nogpu.f90:    subroutine register_GPU(f)
src/nogpu.f90:    endsubroutine register_GPU
src/nogpu.f90:    subroutine finalize_GPU
src/nogpu.f90:    endsubroutine finalize_GPU
src/nogpu.f90:    subroutine rhs_GPU(f,itsub,early_finalize)
src/nogpu.f90:    endsubroutine rhs_GPU
src/nogpu.f90:    function get_ptr_GPU(ind1,ind2,lout) result(pFarr)
src/nogpu.f90:    endfunction get_ptr_GPU
src/nogpu.f90:    subroutine copy_farray_from_GPU(f)
src/nogpu.f90:    endsubroutine copy_farray_from_GPU
src/nogpu.f90:endmodule  GPU
src/gpu_astaroth.cu:/*                             gpu_astaroth_ansi.cu
src/gpu_astaroth.cu://CUDA libraries
src/gpu_astaroth.cu:/*cudaError_t checkErr(cudaError_t result) {
src/gpu_astaroth.cu:  if (result != cudaSuccess) {
src/gpu_astaroth.cu:    fprintf(stderr, "CUDA Runtime Error: %s \n", 
src/gpu_astaroth.cu:            cudaGetErrorString(result));
src/gpu_astaroth.cu:    assert(result == cudaSuccess);
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NX, &nx, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NY, &ny, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NZ, &nz, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_PAD_SIZE, &pad_size, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BOUND_SIZE, &bound_size, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_X, &comp_domain_size_x, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Y, &comp_domain_size_y, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Z, &comp_domain_size_z, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NELEMENTS_FLOAT, &nelements_float, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_X, &domain_size_x, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Y, &domain_size_y, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Z, &domain_size_z, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Y_OFFSET, &h_w_grid_y_offset, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Z_OFFSET, &h_w_grid_z_offset, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Y_OFFSET, &h_grid_y_offset, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Z_OFFSET, &h_grid_z_offset, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_TOP, &cx_top, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_TOP, &cy_top, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_TOP, &cz_top, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_BOT, &cx_bot, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_BOT, &cy_bot, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_BOT, &cz_bot, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DX, &dx, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DY, &dy, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DZ, &dz, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_XORIG, &xorig, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_YORIG, &yorig, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_ZORIG, &zorig, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_INTERP_ORDER, &interp_order, sizeof(int)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_Q_SHEAR, &q_shear, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_OMEGA, &omega, sizeof(float)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LFORCING, &lforcing, sizeof(int)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LSHEAR, &lshear, sizeof(int)) );
src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LCORIOLIS, &lcoriolis, sizeof(int)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA1, &h_ALPHA1, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA2, &h_ALPHA2, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA3, &h_ALPHA3, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA1, &h_BETA1, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA2, &h_BETA2, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA3, &h_BETA3, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NU_VISC, &nu_visc, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CS2_SOUND, &cs2_sound, sizeof(float)) );	
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_9, &flt_9, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_45, &flt_45, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_60, &flt_60, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_2, &flt_2, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_27, &flt_27, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_270, &flt_270, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_490, &flt_490, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_180, &flt_180, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(float)) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(float)) );
src/gpu_astaroth.cu:bool finalizeGpu(float *uu_x, float *uu_y, float *uu_z, float *lnrho){
src/gpu_astaroth.cu:/* Frees memory allocated on GPU.
src/gpu_astaroth.cu:	checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/gpu_astaroth.cu:	//cudaEventDestroy( start );
src/gpu_astaroth.cu:	//cudaEventDestroy( stop );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_lnrho) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_x) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_y) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_z) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_umax) ); checkErr( cudaFree(d_umin) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_urms) );
src/gpu_astaroth.cu: 	checkErr( cudaFree(d_uxrms) ); checkErr( cudaFree(d_uyrms) ); checkErr( cudaFree(d_uzrms) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_rhorms) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_rhomax) ); checkErr( cudaFree(d_rhomin) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_uxmax) ); checkErr( cudaFree(d_uymax) ); checkErr( cudaFree(d_uzmax) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_uxmin) ); checkErr( cudaFree(d_uymin) ); checkErr( cudaFree(d_uzmin) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_partial_result) );
src/gpu_astaroth.cu:	checkErr( cudaFree(d_halo) );
src/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_lnrho) );
src/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu) );
src/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_x) );
src/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_y) );
src/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_z) );*/
src/gpu_astaroth.cu:	cudaDeviceReset();
src/gpu_astaroth.cu:	rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, d_w_lnrho, d_w_uu_x, d_w_uu_y, d_w_uu_z, d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest);
src/gpu_astaroth.cu://void intitializeGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho, int nx, int ny, int nz, int nghost, float *x, float *y, float *z, float NU_VISC, float cs2_sound){ 
src/gpu_astaroth.cu:void intitializeGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho, int nx, int ny, int nz, int nghost, float *x, float *y, float *z, float nu, float cs2){ 
src/gpu_astaroth.cu:		cudaGetDevice(&device);
src/gpu_astaroth.cu:		//cudaSetDevice(device); //Not yet enabled
src/gpu_astaroth.cu:		cudaDeviceReset();
src/gpu_astaroth.cu:		checkErr(cudaMalloc ((void **) &d_halo, sizeof(float)*halo_size));
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_lnrho, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_x, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_y, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_z, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_lnrho, sizeof(float)*W_GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_x, sizeof(float)*W_GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_y, sizeof(float)*W_GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_z, sizeof(float)*W_GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_lnrho_dest, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_x_dest, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_y_dest, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_z_dest, sizeof(float)*GRID_SIZE) );
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_umax, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_umin, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_urms, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxrms, sizeof(float)) );  //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uyrms, sizeof(float)) );  //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzrms, sizeof(float)) );  //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhorms, sizeof(float)) ); //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhomax, sizeof(float)) ); //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhomin, sizeof(float)) ); //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxmax, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxmin, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uymax, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uymin, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzmax, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzmin, sizeof(float)) );   //TODO this somewhere else
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_lnrho_dest, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_x_dest, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_y_dest, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_z_dest, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/equ.f90:!  14-feb-17/MR: adaptations for use of GPU kernels in calculating the rhss of the pde
src/equ.f90:      use Gpu
src/equ.f90:                     lyinyang .or. lgpu .or. &   !!!
src/equ.f90:      if (lgpu) then
src/equ.f90:        call rhs_gpu(f,itsub,early_finalize)
src/equ.f90:          call copy_farray_from_GPU(f)
src/equ.f90:      !if (lgpu) then
src/equ.f90:      !  call freeze_gpu
src/gpu.sed:s/^ *GPU *= *\([A-Za-z0-9_]*\) *$/\U\1=\U\1/
src/gpu.sed:w CUDA_MAKEDIR/PC_modulesources.h
src/test_methods/testfield_xz.f90:  real, dimension(nx)              :: cx,sx                     !GPU => DEVICE
src/test_methods/testfield_xz.f90:  real, dimension(nx,nz,3,3)       :: Minv                      !GPU => DEVICE
src/cuda/gpu_astaroth.cu:/*                             gpu_astaroth_ansi.cu
src/cuda/gpu_astaroth.cu://CUDA libraries
src/cuda/gpu_astaroth.cu:/*cudaError_t checkErr(cudaError_t result) {
src/cuda/gpu_astaroth.cu:  if (result != cudaSuccess) {
src/cuda/gpu_astaroth.cu:    fprintf(stderr, "CUDA Runtime Error: %s \n", 
src/cuda/gpu_astaroth.cu:            cudaGetErrorString(result));
src/cuda/gpu_astaroth.cu:    assert(result == cudaSuccess);
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NX, &nx, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NY, &ny, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NZ, &nz, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_PAD_SIZE, &pad_size, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BOUND_SIZE, &bound_size, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_X, &comp_domain_size_x, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Y, &comp_domain_size_y, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Z, &comp_domain_size_z, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NELEMENTS_FLOAT, &nelements_float, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_X, &domain_size_x, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Y, &domain_size_y, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Z, &domain_size_z, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Y_OFFSET, &h_w_grid_y_offset, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Z_OFFSET, &h_w_grid_z_offset, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Y_OFFSET, &h_grid_y_offset, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Z_OFFSET, &h_grid_z_offset, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_TOP, &cx_top, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_TOP, &cy_top, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_TOP, &cz_top, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_BOT, &cx_bot, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_BOT, &cy_bot, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_BOT, &cz_bot, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DX, &dx, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DY, &dy, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DZ, &dz, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_XORIG, &xorig, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_YORIG, &yorig, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_ZORIG, &zorig, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_INTERP_ORDER, &interp_order, sizeof(int)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_Q_SHEAR, &q_shear, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_OMEGA, &omega, sizeof(float)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LFORCING, &lforcing, sizeof(int)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LSHEAR, &lshear, sizeof(int)) );
src/cuda/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LCORIOLIS, &lcoriolis, sizeof(int)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA1, &h_ALPHA1, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA2, &h_ALPHA2, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA3, &h_ALPHA3, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA1, &h_BETA1, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA2, &h_BETA2, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA3, &h_BETA3, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NU_VISC, &nu_visc, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CS2_SOUND, &cs2_sound, sizeof(float)) );	
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_9, &flt_9, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_45, &flt_45, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_60, &flt_60, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_2, &flt_2, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_27, &flt_27, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_270, &flt_270, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_490, &flt_490, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_180, &flt_180, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(float)) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(float)) );
src/cuda/gpu_astaroth.cu:bool finalizeGpu(float *uu_x, float *uu_y, float *uu_z, float *lnrho){
src/cuda/gpu_astaroth.cu:/* Frees memory allocated on GPU.
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/gpu_astaroth.cu:	//cudaEventDestroy( start );
src/cuda/gpu_astaroth.cu:	//cudaEventDestroy( stop );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_lnrho) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_x) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_y) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_uu_z) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_umax) ); checkErr( cudaFree(d_umin) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_urms) );
src/cuda/gpu_astaroth.cu: 	checkErr( cudaFree(d_uxrms) ); checkErr( cudaFree(d_uyrms) ); checkErr( cudaFree(d_uzrms) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_rhorms) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_rhomax) ); checkErr( cudaFree(d_rhomin) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_uxmax) ); checkErr( cudaFree(d_uymax) ); checkErr( cudaFree(d_uzmax) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_uxmin) ); checkErr( cudaFree(d_uymin) ); checkErr( cudaFree(d_uzmin) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_partial_result) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFree(d_halo) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_lnrho) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_x) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_y) );
src/cuda/gpu_astaroth.cu:	checkErr( cudaFreeHost(slice_uu_z) );*/
src/cuda/gpu_astaroth.cu:	cudaDeviceReset();
src/cuda/gpu_astaroth.cu:	rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, d_w_lnrho, d_w_uu_x, d_w_uu_y, d_w_uu_z, d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest);
src/cuda/gpu_astaroth.cu://void intitializeGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho, int nx, int ny, int nz, int nghost, float *x, float *y, float *z, float NU_VISC, float cs2_sound){ 
src/cuda/gpu_astaroth.cu:void intitializeGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho, int nx, int ny, int nz, int nghost, float *x, float *y, float *z, float nu, float cs2){ 
src/cuda/gpu_astaroth.cu:		cudaGetDevice(&device);
src/cuda/gpu_astaroth.cu:		//cudaSetDevice(device); //Not yet enabled
src/cuda/gpu_astaroth.cu:		cudaDeviceReset();
src/cuda/gpu_astaroth.cu:		checkErr(cudaMalloc ((void **) &d_halo, sizeof(float)*halo_size));
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_lnrho, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_x, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_y, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_z, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_lnrho, sizeof(float)*W_GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_x, sizeof(float)*W_GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_y, sizeof(float)*W_GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_w_uu_z, sizeof(float)*W_GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_lnrho_dest, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_x_dest, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_y_dest, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc(&d_uu_z_dest, sizeof(float)*GRID_SIZE) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_umax, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_umin, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_urms, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uyrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhorms, sizeof(float)) ); //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhomax, sizeof(float)) ); //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_rhomin, sizeof(float)) ); //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxmax, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uxmin, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uymax, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uymin, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzmax, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMalloc((float**) &d_uzmin, sizeof(float)) );   //TODO this somewhere else
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_lnrho_dest, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_x_dest, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_y_dest, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/gpu_astaroth.cu:		checkErr( cudaMemcpy(d_uu_z_dest, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) ); 
src/cuda/src_new/gpu_astaroth.cc:/*                             gpu_astaroth_ansi.cu
src/cuda/src_new/gpu_astaroth.cc:basic functions for accessing the GPU interface
src/cuda/src_new/gpu_astaroth.cc://GPU interface
src/cuda/src_new/gpu_astaroth.cc:#include "gpu/gpu.h"
src/cuda/src_new/gpu_astaroth.cc:#include "gpu/cuda/cuda_generic.cuh"
src/cuda/src_new/gpu_astaroth.cc://Do the 'isubstep'th integration step on all GPUs on the node and handle boundaries
src/cuda/src_new/gpu_astaroth.cc://TODO: note, isubstep starts from 0 on the GPU side (i.e. rk3 does substeps 0, 1, 2)
src/cuda/src_new/gpu_astaroth.cc:extern "C" void substepGPU(int isubstep, bool full=false)
src/cuda/src_new/gpu_astaroth.cc:        GPUUpdateForcingCoefs(&forcing_params);		// load into GPU
src/cuda/src_new/gpu_astaroth.cc:        GPULoad(&h_grid); 
src/cuda/src_new/gpu_astaroth.cc:        //!!!GPULoadOuterHalos(&h_grid,halo_buffer);
src/cuda/src_new/gpu_astaroth.cc:        GPULoad(&h_grid); 
src/cuda/src_new/gpu_astaroth.cc:        //exchange_halos_cuda_generic(false);      // no circular halo exchange
src/cuda/src_new/gpu_astaroth.cc:    //Integrate on the GPUs in this node
src/cuda/src_new/gpu_astaroth.cc:    GPUIntegrateStep(isubstep-1, dt);
src/cuda/src_new/gpu_astaroth.cc:    //!!!GPUStoreInternalHalos(&h_grid,halo_buffer);
src/cuda/src_new/gpu_astaroth.cc:    GPUStore(&h_grid);
src/cuda/src_new/gpu_astaroth.cc://CRASH("GPUDiagnostics");
src/cuda/src_new/gpu_astaroth.cc:extern "C" void registerGPU(real* farray)
src/cuda/src_new/gpu_astaroth.cc://Setup the GPUs in the node to be ready for computation
src/cuda/src_new/gpu_astaroth.cc:extern "C" void initGPU()
src/cuda/src_new/gpu_astaroth.cc:    //Initialize GPUs in the node
src/cuda/src_new/gpu_astaroth.cc:    GPUSelectImplementation(CUDA_FOR_PENCIL);
src/cuda/src_new/gpu_astaroth.cc:    GPUInit(&cparams, &run_params);         //Allocs memory on the GPU and loads device constants
src/cuda/src_new/gpu_astaroth.cc:extern "C" void initializeGPU()
src/cuda/src_new/gpu_astaroth.cc:    //Setup configurations used for initializing and running the GPU code
src/cuda/src_new/gpu_astaroth.cc:        GPUInitialize(&cparams, &run_params, h_grid);         // loads device constants
src/cuda/src_new/gpu_astaroth.cc:                GPUStore(&h_grid);
src/cuda/src_new/gpu_astaroth.cc://Destroy the GPUs in the node (not literally hehe)
src/cuda/src_new/gpu_astaroth.cc:extern "C" void finalizeGPU()
src/cuda/src_new/gpu_astaroth.cc:    //Deallocate everything on the GPUs and reset
src/cuda/src_new/gpu_astaroth.cc:    GPUDestroy();
src/cuda/src_new/howtoadd_makefiledepend_to_cmakelists.txt:Replace CUDA_ADD_LIBRARY command in src_new/CMakeLists.txt near line 204 with:
src/cuda/src_new/howtoadd_makefiledepend_to_cmakelists.txt:CUDA_ADD_LIBRARY(astaroth_core SHARED gpu ${CUDA_MODULES} common ${HEADER_FILES} OPTIONS --compiler-options "-fpic")
src/cuda/src_new/Makefile.depend:gpu_astaroth.o: gpu_astaroth.cc $(COMMON_HEADERS) common/PC_moduleflags.h common/PC_module_parfuncs.h common/forcing.h gpu/gpu.h gpu/cuda/cuda_generic.cuh diagnostics/diagnostics.h $(CHEADERS)
src/cuda/src_new/Makefile.depend:gpu/gpu.o: gpu/gpu.cc gpu/gpu.h common/errorhandler.h gpu/cuda/cuda_generic.cuh gpu/cuda/core/concur_cuda_core.cuh gpu/cuda/generic/collectiveops_cuda_generic.cuh common/datatypes.h common/config.h common/grid.h common/slice.h common/forcing.h 
src/cuda/src_new/Makefile.depend:gpu/cuda/cuda_generic.o: gpu/cuda/cuda_generic.cu gpu/cuda/cuda_generic.cuh utils/utils.h gpu/cuda/core/cuda_core.cuh gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh gpu/cuda/core/concur_cuda_core.cuh gpu/cuda/core/copyHalosConcur.cuh gpu/cuda/generic/rk3_cuda_generic.cuh gpu/cuda/generic/boundcond_cuda_generic.cuh gpu/cuda/generic/slice_cuda_generic.cuh gpu/cuda/generic/collectiveops_cuda_generic.cuh common/datatypes.h common/config.h common/grid.h common/slice.h common/forcing.h 
src/cuda/src_new/Makefile.depend:diagnostics/timeseries_diagnostics.o: diagnostics/timeseries_diagnostics.cc common/grid.h common/qualify.h utils/utils.h gpu/cuda/cuda_generic.cuh ../../cparam_c.h ../../cdata_c.h ../../diagnostics_c.h diagnostics/PC_module_diagfuncs.h diagnostics/PC_modulediags_init.h diagnostics/PC_modulediags.h
src/cuda/src_new/Makefile.depend:gpu/cuda/generic/boundcond_cuda_generic.o: gpu/cuda/generic/boundcond_cuda_generic.cu gpu/cuda/generic/boundcond_cuda_generic.cuh gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh common/config.h common/grid.h common/errorhandler.h
src/cuda/src_new/Makefile.depend:gpu/cuda/generic/collectiveops_cuda_generic.o: gpu/cuda/generic/collectiveops_cuda_generic.cu gpu/cuda/generic/collectiveops_cuda_generic.cuh gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh utils/utils.h common/errorhandler.h common/config.h
src/cuda/src_new/Makefile.depend:gpu/cuda/generic/rk3_cuda_generic.o: gpu/cuda/generic/rk3_cuda_generic.cu gpu/cuda/generic/rk3_cuda_generic.cuh gpu/cuda/generic/diff_cuda_generic.cuh gpu/cuda/core/errorhandler_cuda.cuh common/datatypes.h common/defines.h common/errorhandler.h common/config.h common/grid.h common/PC_moduleflags.h
src/cuda/src_new/Makefile.depend:gpu/cuda/generic/slice_cuda_generic.o: gpu/cuda/generic/slice_cuda_generic.cu gpu/cuda/generic/slice_cuda_generic.cuh common/config.h common/grid.h common/errorhandler.h common/slice.h gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh common/PC_moduleflags.h
src/cuda/src_new/Makefile.depend:gpu/cuda/core/concur_cuda_core.o: gpu/cuda/core/concur_cuda_core.cu gpu/cuda/core/concur_cuda_core.cuh
src/cuda/src_new/Makefile.depend:gpu/cuda/core/copyHalosConcur.o: gpu/cuda/core/copyHalosConcur.cu gpu/cuda/core/copyHalosConcur.cuh gpu/cuda/cuda_generic.cuh common/PC_moduleflags.h common/PC_modulepardecs.h common/datatypes.h common/errorhandler.h gpu/cuda/core/dconsts_core.cuh ../../cdata_c.h
src/cuda/src_new/Makefile.depend:gpu/cuda/core/cuda_core.o: gpu/cuda/core/cuda_core.cu gpu/cuda/core/cuda_core.cuh gpu/cuda/cuda_generic.cuh common/datatypes.h common/errorhandler.h gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh gpu/cuda/core/copyHalosConcur.cuh common/PC_moduleflags.h common/PC_modulepardecs.h common/config.h common/grid.h common/slice.h common/forcing.h common/PC_moduleflags.h common/PC_modulepars.h common/PC_module_parfuncs.h
src/cuda/src_new/Makefile:CUSOURCES = $(wildcard gpu/cuda/*.cu) $(wildcard gpu/cuda/generic/*.cu) $(wildcard gpu/cuda/core/*.cu)
src/cuda/src_new/Makefile:CCSOURCES = $(wildcard gpu/*.cc) $(wildcard common/*.cc) $(wildcard diagnostics/*.cc)
src/cuda/src_new/Makefile:MAIN_SRC = gpu_astaroth.cc
src/cuda/src_new/Makefile:MAIN_OBJ = gpu_astaroth.o
src/cuda/src_new/Makefile:# Settings for taito-gpu
src/cuda/src_new/Makefile:#  1) cuda/9.0   3) openmpi/2.0.1_ic16.0            5) hdf5/1.8.16_openmpi_2.0.1_ic16.0 7) gcc/5.3.0
src/cuda/src_new/Makefile:#  2) intel/2016 4) fftw/2.1.5_openmpi_2.0.1_ic16.0 6) cuda/9.1
src/cuda/src_new/Makefile:ENVIRON = -D MODPRE=${MODULE_PREFIX} -D MODIN=${MODULE_INFIX} -D MODSUF=${MODULE_SUFFIX} -DGPU_ASTAROTH
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:#include "gpu/cuda/cuda_generic.cuh"
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:// Contains declarations if idiag_* variables and definition of function init_diagnostics tb called by gpu_astaroth.
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:  // Calculate and save all of the diagnostic variables calculated within the CUDA devices. 
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:  // Contains automatically generated calls to reduce_cuda_PC according to required diagnostics of the different modules.
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:    diag=reduce_cuda_PC(ldensity_nolog ? SUM_SCAL : SUM_EXP, h_grid.LNRHO)*box_volume/nw;
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:                        diag=reduce_cuda_PC(MAX_SCAL, h_grid.LNRHO);
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:                        diag=reduce_cuda_PC(MIN_SCAL, h_grid.LNRHO);
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:                        diag=reduce_cuda_PC(ldensity_nolog ? SUM_SCAL : SUM_EXP, h_grid.LNRHO);
src/cuda/src_new/diagnostics/timeseries_diagnostics.cc:                        diag=reduce_cuda_PC(ldensity_nolog ? RMS_SCAL : RMS_EXP, h_grid.LNRHO);
src/cuda/src_new/diagnostics/diagnostics.h:// only defined & used if GPU_ASTAROTH is defined
src/cuda/src_new/common/datatypes.h:#ifdef GPU_ASTAROTH
src/cuda/src_new/common/forcing.h:#ifdef GPU_ASTAROTH
src/cuda/src_new/common/slice.h:#ifdef GPU_ASTAROTH
src/cuda/src_new/common/defines.h://  b) Calling device functions with dynamically allocated memory is wonky (CUDA 6)
src/cuda/src_new/common/defines.h:#ifdef GPU_ASTAROTH
src/cuda/src_new/common/defines_dims_PC.h://  b) Calling device functions with dynamically allocated memory is wonky (CUDA 6)
src/cuda/src_new/CMakeLists.txt:option(CUDA_BUILD_LEGACY "Builds GPU code for older GPUs (Fermi)" OFF)
src/cuda/src_new/CMakeLists.txt:#CUDA
src/cuda/src_new/CMakeLists.txt:find_package(CUDA)
src/cuda/src_new/CMakeLists.txt:if (NOT CUDA_FOUND)
src/cuda/src_new/CMakeLists.txt:    #find_package(CUDA REQUIRED) gives a confusing error message if it fails, 
src/cuda/src_new/CMakeLists.txt:    message(FATAL_ERROR "CUDA not found")
src/cuda/src_new/CMakeLists.txt:#set(CUDA_VERBOSE_BUILD OFF)
src/cuda/src_new/CMakeLists.txt:#CUDA settings
src/cuda/src_new/CMakeLists.txt:set(CUDA_SEPARABLE_COMPILATION ON)
src/cuda/src_new/CMakeLists.txt:set(CUDA_PROPAGATE_HOST_FLAGS ON)
src/cuda/src_new/CMakeLists.txt:#set(CUDA_BUILD_CUBIN ON) #Requires that we're compiling for only one architecture
src/cuda/src_new/CMakeLists.txt:#----------------------Setup CUDA compilation flags----------------------------#
src/cuda/src_new/CMakeLists.txt:set(CUDA_ARCH_FLAGS -gencode arch=compute_35,code=sm_35;
src/cuda/src_new/CMakeLists.txt:#Enable support for Pascal and Volta GPUs
src/cuda/src_new/CMakeLists.txt:if (CUDA_VERSION_MAJOR GREATER 8 OR CUDA_VERSION_MAJOR EQUAL 8)
src/cuda/src_new/CMakeLists.txt:    set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS};
src/cuda/src_new/CMakeLists.txt:    if (CUDA_VERSION_MAJOR GREATER 9 OR CUDA_VERSION_MAJOR EQUAL 9)
src/cuda/src_new/CMakeLists.txt:        set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS};
src/cuda/src_new/CMakeLists.txt:        message(WARNING "CUDA 9.0 or greater not available, \
src/cuda/src_new/CMakeLists.txt:                         cannot generate code for cc 7.0 GPUs")
src/cuda/src_new/CMakeLists.txt:    message(WARNING "CUDA 8.0 or greater not available, \
src/cuda/src_new/CMakeLists.txt:                     cannot generate code for cc 6.0 GPUs")
src/cuda/src_new/CMakeLists.txt:#Generate code for older GPUs
src/cuda/src_new/CMakeLists.txt:if (CUDA_BUILD_LEGACY)
src/cuda/src_new/CMakeLists.txt:    set(CUDA_ARCH_FLAGS -gencode arch=compute_20,code=sm_20)
src/cuda/src_new/CMakeLists.txt:#Additional CUDA optimization flags
src/cuda/src_new/CMakeLists.txt:    #Doesn't set any additional flags, see CUDA_NVCC_FLAGS_DEBUG how to add more
src/cuda/src_new/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE}) 
src/cuda/src_new/CMakeLists.txt:#Additional CUDA debug flags
src/cuda/src_new/CMakeLists.txt:    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG};
src/cuda/src_new/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH_FLAGS}")
src/cuda/src_new/CMakeLists.txt:    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_ARCH_FLAGS}")
src/cuda/src_new/CMakeLists.txt:message("CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})
src/cuda/src_new/CMakeLists.txt:add_subdirectory (gpu)
src/cuda/src_new/CMakeLists.txt:#CUDA sources
src/cuda/src_new/CMakeLists.txt:set(CUDA_SRC_DIR gpu/cuda)
src/cuda/src_new/CMakeLists.txt:set(CUDA_CORE ${CUDA_SRC_DIR}/core/cuda_core.cu)
src/cuda/src_new/CMakeLists.txt:set(CUDA_GENERIC
src/cuda/src_new/CMakeLists.txt:     ${CUDA_SRC_DIR}/generic/slice_cuda_generic.cu 
src/cuda/src_new/CMakeLists.txt:     ${CUDA_SRC_DIR}/generic/boundcond_cuda_generic.cu  
src/cuda/src_new/CMakeLists.txt:     ${CUDA_SRC_DIR}/generic/rk3_cuda_generic.cu
src/cuda/src_new/CMakeLists.txt:     ${CUDA_SRC_DIR}/generic/collectiveops_cuda_generic.cu
src/cuda/src_new/CMakeLists.txt:     ${CUDA_SRC_DIR}/cuda_generic.cu)
src/cuda/src_new/CMakeLists.txt:set(CUDA_MODULES ${CUDA_CORE} ${CUDA_GENERIC})
src/cuda/src_new/CMakeLists.txt:#Create shared library of the GPU module (astaroth_core)
src/cuda/src_new/CMakeLists.txt:CUDA_ADD_LIBRARY(astaroth_core SHARED gpu ${CUDA_MODULES} common OPTIONS --compiler-options "-fpic")
src/cuda/src_new/CMakeLists.txt:    cuda_add_executable(ac_run main.cc)
src/cuda/src_new/CMakeLists.txt:    target_link_libraries(ac_run utils gpu common cpu_model astaroth_core)
src/cuda/src_new/gpu_astaroth_new.cc:/*                             gpu_astaroth_ansi.cu
src/cuda/src_new/gpu_astaroth_new.cc:basic functions for accessing the GPU interface
src/cuda/src_new/gpu_astaroth_new.cc://GPU interface
src/cuda/src_new/gpu_astaroth_new.cc:#include "gpu/gpu.h"
src/cuda/src_new/gpu_astaroth_new.cc://Do the 'isubstep'th integration step on all GPUs on the node and handle boundaries
src/cuda/src_new/gpu_astaroth_new.cc://TODO: note, isubstep starts from 0 on the GPU side (i.e. rk3 does substeps 0, 1, 2)
src/cuda/src_new/gpu_astaroth_new.cc:    //GPUStoreInnerHalos(host_grid) //copyinnerhalostohost
src/cuda/src_new/gpu_astaroth_new.cc:    //GPULoadOuterHalos(host_grid) //copyouterhalostodevice
src/cuda/src_new/gpu_astaroth_new.cc:        GPULoadForcingParams(&forcing_params);
src/cuda/src_new/gpu_astaroth_new.cc:    //Integrate on the GPUs in this node
src/cuda/src_new/gpu_astaroth_new.cc:    GPUIntegrateStep(isubstep, dt);
src/cuda/src_new/gpu_astaroth_new.cc://Setup configuration structs used for initializing and running the GPU code
src/cuda/src_new/gpu_astaroth_new.cc://Setup the GPUs in the node to be ready for computation
src/cuda/src_new/gpu_astaroth_new.cc:void intitializeGPU(real *uu_x, real *uu_y, real *uu_z, real *lnrho, 
src/cuda/src_new/gpu_astaroth_new.cc:    GPULoadForcingParams(&forcing_params);
src/cuda/src_new/gpu_astaroth_new.cc:    //Initialize GPUs in the node
src/cuda/src_new/gpu_astaroth_new.cc:    GPUSelectImplementation(CUDA_GENERIC);
src/cuda/src_new/gpu_astaroth_new.cc:    GPUInit(&cparams, &run_params); //Allocs memory on the GPU and loads device constants
src/cuda/src_new/gpu_astaroth_new.cc:    GPULoad(&grid); //Loads the whole grid from host to device
src/cuda/src_new/gpu_astaroth_new.cc:    //TODO: Any optional steps, for example store the first GPU slice to slice arrays on host:
src/cuda/src_new/gpu_astaroth_new.cc:    //GPUGetSlice(slice_lnrho, slice_uu, slice_uu_x, slice_uu_y, slice_uu_z);
src/cuda/src_new/gpu_astaroth_new.cc://Destroy the GPUs in the node (not literally hehe)
src/cuda/src_new/gpu_astaroth_new.cc:bool finalizeGPU(real *uu_x, real *uu_y, real *uu_z, real *lnrho)
src/cuda/src_new/gpu_astaroth_new.cc:    //Deallocate everything on the GPUs and reset
src/cuda/src_new/gpu_astaroth_new.cc:    GPUDestroy();
src/cuda/src_new/gpu_astaroth_new.cc:    //TODO: Deallocate all host memory possibly allocated in initializeGPU() etc
src/cuda/src_new/gpu_astaroth_new.cc:    //initializeGPU(TODO the appropriate params here);
src/cuda/src_new/gpu_astaroth_new.cc:    //finalizeGPU(TODO the appropriate params here);
src/cuda/src_new/gpu/gpu.cc:*   Sets the GPU interface up by pointing the interface functions to the functions
src/cuda/src_new/gpu/gpu.cc:*	For example: only GPUInit() is visible to someone who includes "gpu/gpu.h". 
src/cuda/src_new/gpu/gpu.cc:*	GPUInit() is set here to point to some actual implementation, for example
src/cuda/src_new/gpu/gpu.cc:*	init_cuda_generic() defined in "gpu/cuda/cuda_generic.cu". 	
src/cuda/src_new/gpu/gpu.cc:*	interface functions do not have to care whether we use, say, cuda, opencl, 
src/cuda/src_new/gpu/gpu.cc:*	multi-GPU or some alternative boundary condition function. Everything
src/cuda/src_new/gpu/gpu.cc:*	Therefore even if we make drastic changes in the GPU code, we don't have to
src/cuda/src_new/gpu/gpu.cc:#include "gpu.h"
src/cuda/src_new/gpu/gpu.cc:#include "cuda/cuda_generic.cuh"
src/cuda/src_new/gpu/gpu.cc:GPUInitFunc               GPUInit               = &init_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUInitializeFunc         GPUInitialize         = &initialize_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUDestroyFunc            GPUDestroy            = &destroy_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPULoadFunc               GPULoad               = &load_grid_cuda_generic;  //Load from host to device
src/cuda/src_new/gpu/gpu.cc:GPUStoreFunc              GPUStore              = &store_grid_cuda_generic; //Store from device to host
src/cuda/src_new/gpu/gpu.cc:GPULoadOuterHalosFunc     GPULoadOuterHalos     = &load_outer_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUStoreInternalHalosFunc GPUStoreInternalHalos = &store_internal_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.cc:GPUUpdateForcingCoefsFunc GPUUpdateForcingCoefs = &update_forcing_coefs_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPULoadForcingParamsFunc  GPULoadForcingParams  = &load_forcing_params_cuda_generic;
src/cuda/src_new/gpu/gpu.cc://GPU solver interface
src/cuda/src_new/gpu/gpu.cc:GPUIntegrateFunc     GPUIntegrate     = &integrate_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUIntegrateStepFunc GPUIntegrateStep = &integrate_step_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUBoundcondStepFunc GPUBoundcondStep = &boundcond_step_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:GPUReduceFunc    GPUReduce   = NULL;
src/cuda/src_new/gpu/gpu.cc:GPUGetSliceFunc  GPUGetSlice = &get_slice_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:const char* impl_type_names[] = {"CUDA_GENERIC",
src/cuda/src_new/gpu/gpu.cc:                                 "CUDA_19P",
src/cuda/src_new/gpu/gpu.cc:                                 "CUDA_55P",
src/cuda/src_new/gpu/gpu.cc:                                 "CUDA_MAXWELL",
src/cuda/src_new/gpu/gpu.cc:                                 "CUDA_FOR_PENCIL"};
src/cuda/src_new/gpu/gpu.cc:    void GPUSelectImplementation(ImplType type) 
src/cuda/src_new/gpu/gpu.cc:        if (type != CUDA_GENERIC) {
src/cuda/src_new/gpu/gpu.cc:            printf("Warning, tried to select some other implementation than CUDA_GENERIC, this has no effect since DO_MINIMAL_BUILD is set.\n");
src/cuda/src_new/gpu/gpu.cc:    //#include "cuda/cuda_19p.cuh"
src/cuda/src_new/gpu/gpu.cc:    //#include "cuda/cuda_55p.cuh"
src/cuda/src_new/gpu/gpu.cc:    //#include "cuda/cuda_maxwell.cuh"
src/cuda/src_new/gpu/gpu.cc:    //Select the GPU implementation (yes, this could be done much more nicely
src/cuda/src_new/gpu/gpu.cc:    void GPUSelectImplementation(ImplType type) {
src/cuda/src_new/gpu/gpu.cc:        if (type == CUDA_GENERIC) {
src/cuda/src_new/gpu/gpu.cc:            GPUInit                 = &init_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUDestroy              = &destroy_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPULoad                 = &load_grid_cuda_generic;  //Load from host to device
src/cuda/src_new/gpu/gpu.cc:            GPUStore                = &store_grid_cuda_generic; //Store from device to host
src/cuda/src_new/gpu/gpu.cc:#ifndef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.cc:            GPULoadForcingParams    = &load_forcing_params_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPULoadOuterHalos       = &load_outer_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUStoreInternalHalos   = &store_internal_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            //GPU solver interface
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrate     = &integrate_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUBoundcondStep = &boundcond_step_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrateStep = &integrate_step_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:#ifndef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.cc:            GPUReduce        = &reduce_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUGetSlice = &get_slice_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:        }/*else if (type == CUDA_19P) {
src/cuda/src_new/gpu/gpu.cc:            GPUInit    = &init_cuda_19p;
src/cuda/src_new/gpu/gpu.cc:            GPUDestroy = &destroy_cuda_19p;
src/cuda/src_new/gpu/gpu.cc:            GPULoad    = &load_grid_cuda_19p;  //Load from host to device
src/cuda/src_new/gpu/gpu.cc:            GPUStore   = &store_grid_cuda_19p; //Store from device to host
src/cuda/src_new/gpu/gpu.cc:            GPULoadForcingParams = NULL;
src/cuda/src_new/gpu/gpu.cc:            //GPU solver interface
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrate     = &integrate_cuda_19p;
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrateStep = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUBoundcondStep = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUReduce        = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUGetSlice = &get_slice_cuda_19p;
src/cuda/src_new/gpu/gpu.cc:        } else if (type == CUDA_55P) {
src/cuda/src_new/gpu/gpu.cc:            GPUInit    = &init_cuda_55p;
src/cuda/src_new/gpu/gpu.cc:            GPUDestroy = &destroy_cuda_55p;
src/cuda/src_new/gpu/gpu.cc:            GPULoad    = &load_grid_cuda_55p;  //Load from host to device
src/cuda/src_new/gpu/gpu.cc:            GPUStore   = &store_grid_cuda_55p; //Store from device to host
src/cuda/src_new/gpu/gpu.cc:            GPULoadForcingParams = NULL;
src/cuda/src_new/gpu/gpu.cc:            //GPU solver interface
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrate     = &integrate_cuda_55p;
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrateStep = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUBoundcondStep = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUReduce        = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUGetSlice = &get_slice_cuda_55p;
src/cuda/src_new/gpu/gpu.cc:        } else if (type == CUDA_MAXWELL) {
src/cuda/src_new/gpu/gpu.cc:            GPUInit              = &init_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPUDestroy           = &destroy_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPULoad              = &load_grid_cuda_maxwell;  //Load from host to device
src/cuda/src_new/gpu/gpu.cc:            GPUStore             = &store_grid_cuda_maxwell; //Store from device to host
src/cuda/src_new/gpu/gpu.cc:            GPULoadForcingParams = &load_forcing_params_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            //GPU solver interface
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrate     = &integrate_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPUBoundcondStep = &boundcond_step_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrateStep = &integrate_step_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPUReduce        = &reduce_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:            GPUGetSlice = &get_slice_cuda_maxwell;
src/cuda/src_new/gpu/gpu.cc:        }*/ else if (type == CUDA_FOR_PENCIL) {
src/cuda/src_new/gpu/gpu.cc:            GPUInit              = &init_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:	    GPUInitialize        = &initialize_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUDestroy           = &destroy_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPULoad              = &load_grid_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUStore             = &store_grid_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:#ifndef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.cc:            GPULoadForcingParams = NULL;
src/cuda/src_new/gpu/gpu.cc:            GPUUpdateForcingCoefs= &update_forcing_coefs_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPULoadOuterHalos    = &load_outer_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            GPUStoreInternalHalos= &store_internal_halos_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:            //GPU solver interface
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrate     = NULL;                   // only substeps called from PC
src/cuda/src_new/gpu/gpu.cc:            GPUBoundcondStep = NULL;     	       // boundary conditions set in PC
src/cuda/src_new/gpu/gpu.cc:            GPUIntegrateStep = &integrate_step_cuda_generic;
src/cuda/src_new/gpu/gpu.cc:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.cc:            GPUReduce        = &reduce_cuda_PC;
src/cuda/src_new/gpu/gpu.cc:            GPUGetSlice = NULL;
src/cuda/src_new/gpu/CMakeLists.txt:add_library (gpu ${SRCS})
src/cuda/src_new/gpu/CMakeLists.txt:target_include_directories (gpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:#include "gpu/cuda/cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void initHaloConcur(GPUContext & ctx, bool lfirstGPU, bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyOxyPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyOxzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyOyzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void synchronizeStreams(const GPUContext & ctx,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void unlockHostMemOuter(const GPUContext & ctx,real* h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void unlockHostMemInner(const GPUContext & ctx,real* h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyIxyPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyIxzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cuh:void copyIyzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cuh:    cudaStream_t streams[NUM_STREAMS];
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cuh:    cudaEvent_t events[NUM_EVENTS];
src/cuda/src_new/gpu/cuda/core/dconsts_core.cuh:*   GPU constants shared among all different implementations.
src/cuda/src_new/gpu/cuda/core/dconsts_core.cuh://Offset for the distance when using multiple GPUs
src/cuda/src_new/gpu/cuda/core/dconsts_core.cuh:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:#include "errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:cudaStream_t strFront;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:void setInnerHalos(const GPUContext & ctx,const int w,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        int iza=(lfirstGPU ? ctx.d_halo_widths_z[bot] : 0)+nghost,
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:            ize=ctx.d_cparams.mz - nghost - (llastGPU ? ctx.d_halo_widths_z[top] : 0);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:void setOuterHalos(const GPUContext & ctx,real* h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void initHaloConcur(GPUContext & ctx, bool lfirstGPU, bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (!lfirstGPU) ctx.d_halo_widths_z[bot]=nghost;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:   	if (!llastGPU) ctx.d_halo_widths_z[top]=nghost;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaStreamCreate(&ctx.d_copy_streams[FRONT]) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaStreamCreate(&ctx.d_copy_streams[BACK]) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaStreamCreate(&ctx.d_copy_streams[BOT]) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaStreamCreate(&ctx.d_copy_streams[TOP]) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaStreamCreate(&ctx.d_copy_streams[LEFTRIGHT]) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaMemcpyToSymbol(d_halo_widths_x, ctx.d_halo_widths_x, 3*sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaMemcpyToSymbol(d_halo_widths_y, ctx.d_halo_widths_y, 3*sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaMemcpyToSymbol(d_halo_widths_z, ctx.d_halo_widths_z, 3*sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        CUDA_ERRCHK( cudaMalloc(&ctx.d_halobuffer,ctx.d_halobuffer_size));            // buffer for yz halos in device
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyOxyPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu://setOuterHalos(ctx,h_grid,lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU){
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	//!!!cudaHostRegister(h_grid, size, cudaHostRegisterDefault);    // time-critical!
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	cudaMemcpyAsync(ctx.d_grid.arr[w], h_grid, size, cudaMemcpyHostToDevice, ctx.d_copy_streams[FRONT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (llastGPU){
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	        //!!!cudaHostRegister(h_grid+h_offset, size, cudaHostRegisterDefault);     // time-critical!
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	cudaMemcpyAsync(ctx.d_grid.arr[w]+d_offset, h_grid+h_offset, size, cudaMemcpyHostToDevice, ctx.d_copy_streams[BACK]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyOxzPlates(const GPUContext & ctx,int w,real * h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        h_offset0=(lfirstGPU ? halo_widths_z[bot] : ctx.start_idx.z)*mxy;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        d_offset0=(lfirstGPU ? halo_widths_z[bot] : 0)*mxy;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        ia=lfirstGPU ? halo_widths_z[bot] : 0; ie=llastGPU ? ctx.d_cparams.mz-halo_widths_z[top] : ctx.d_cparams.mz;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          //!!!cudaHostRegister(h_grid+h_offset, size, cudaHostRegisterDefault);     // time-critical!
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaMemcpyAsync(ctx.d_grid.arr[w]+d_offset, h_grid+h_offset, size, cudaMemcpyHostToDevice, ctx.d_copy_streams[BOT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          //!!!cudaHostRegister(h_grid+h_offset, size, cudaHostRegisterDefault);      // time-critical!
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaMemcpyAsync(ctx.d_grid.arr[w]+d_offset, h_grid+h_offset, size, cudaMemcpyHostToDevice, ctx.d_copy_streams[TOP]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyOyzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        iza = lfirstGPU ? halo_widths_z[bot] : 0;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        ize = ctx.d_cparams.mz - (llastGPU ? halo_widths_z[top] : 0);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        h_offset0=((lfirstGPU ? halo_widths_z[bot] : 0) + ctx.start_idx.z)*mxy + halo_widths_y[bot]*mx;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                        //cudaMemcpyAsync(halo_yz+halo_ind,h_grid+h_offset,lsize,cudaMemcpyHostToHost, ctx.d_copy_streams[LEFTRIGHT]); 
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                        //cudaMemcpyAsync(halo_yz+halo_ind,h_grid+h_offset,rsize,cudaMemcpyHostToHost, ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaMemcpyAsync(ctx.d_halobuffer, halo_yz, ctx.d_halobuffer_size, cudaMemcpyHostToDevice, ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu://  unpacking in global memory; done by GPU kernel in stream LEFTRIGHT
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        int d_offset0=(lfirstGPU ? halo_widths_z[bot] : 0)*mxy + halo_widths_y[bot]*mx;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:cudaMemcpy(&buf,d_grid+offset,3*sizeof(real),cudaMemcpyDeviceToHost);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void synchronizeStreams(const GPUContext & ctx,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU) cudaStreamSynchronize(ctx.d_copy_streams[FRONT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (llastGPU) cudaStreamSynchronize(ctx.d_copy_streams[BACK]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	cudaStreamSynchronize(ctx.d_copy_streams[BOT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	cudaStreamSynchronize(ctx.d_copy_streams[TOP]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaStreamSynchronize(ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void unlockHostMemOuter(const GPUContext & ctx,real* h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU) cudaHostUnregister(h_grid);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (llastGPU){
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:		cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        h_offset0=(lfirstGPU ? halo_widths_z[bot] : ctx.start_idx.z)*mxy;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        ia=lfirstGPU ? halo_widths_z[bot] : 0; ie=llastGPU ? ctx.d_cparams.mz-halo_widths_z[top] : ctx.d_cparams.mz;
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaHostRegister(halo_yz, halo_yz_size, cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaHostUnregister(halo_yz);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void unlockHostMemInner(const GPUContext & ctx,real* h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (lfirstGPU) {
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                  cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        if (llastGPU) {
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                  cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        int iza=(lfirstGPU ? ctx.d_halo_widths_z[bot] : 0)+nghost,
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:            ize=ctx.d_cparams.mz - nghost - (llastGPU ? ctx.d_halo_widths_z[top] : 0);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaHostUnregister(h_grid+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyIxyPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU)    // or kernel?
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu://setInnerHalos(ctx,w,lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	if (lfirstGPU) {
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	  //!!!cudaHostRegister(h_grid+h_offset, px*ny, cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	          cudaMemcpy2DAsync(h_grid+h_offset, px, ctx.d_grid.arr[w]+h_offset, px, sx, ny, 
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                                    cudaMemcpyDeviceToHost, ctx.d_copy_streams[FRONT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	          //cudaMemcpy(h_grid+h_offset,ctx.d_grid.arr[w]+h_offset, sizeof(real)*mxy*nghost,cudaMemcpyDeviceToHost); 
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu://,printf("firstGPU: i, host_offset, dev_offset= %d %d %d \n", i, h_grid+h_offset, ctx.d_grid.arr[w]+h_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	if (llastGPU) {
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	  //!!!cudaHostRegister(h_grid+h_offset, px*ny, cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:	          cudaMemcpy2DAsync(h_grid+h_offset, px, ctx.d_grid.arr[w]+d_offset, px, sx, ny, 
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                                    cudaMemcpyDeviceToHost, ctx.d_copy_streams[BACK]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu://printf("lastGPU: i, host_offset, dev_offset= %d %d %d \n", i, h_grid+h_offset, ctx.d_grid.arr[w]+d_offset);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyIxzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU)    // or __global__?
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        int iza=(lfirstGPU ? ctx.d_halo_widths_z[bot] : 0)+nghost,
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:            ize=ctx.d_cparams.mz - nghost - (llastGPU ? ctx.d_halo_widths_z[top] : 0);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          //!!!cudaHostRegister(h_grid+h_offset, px*halo_widths_y[bot], cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaMemcpy2DAsync(h_grid+h_offset, px, ctx.d_grid.arr[w]+d_offset, px, sx, halo_widths_y[bot], 
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                            cudaMemcpyDeviceToHost,ctx.d_copy_streams[BOT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          //!!!cudaHostRegister(h_grid+h_offset, px*halo_widths_y[top], cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:          cudaMemcpy2DAsync(h_grid+h_offset, px, ctx.d_grid.arr[w]+d_offset, px, sx, halo_widths_y[top],
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                            cudaMemcpyDeviceToHost, ctx.d_copy_streams[TOP]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:__host__ void copyIyzPlates(const GPUContext & ctx,GridType w,real *h_grid,bool lfirstGPU,bool llastGPU)
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        const int iza=(lfirstGPU ? ctx.d_halo_widths_z[bot] : 0)+nghost,
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                  ize=ctx.d_cparams.mz - nghost - (llastGPU ? ctx.d_halo_widths_z[top] : 0);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaMemcpyAsync(halo_yz, ctx.d_halobuffer, halo_size, cudaMemcpyDeviceToHost,ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        //cudaMemcpy(halo_yz, ctx.d_halobuffer, halo_size, cudaMemcpyDeviceToHost);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                        //cudaMemcpyAsync(h_grid+h_offset,halo_yz+halo_ind,lsize,cudaMemcpyHostToHost,ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:                        //cudaMemcpyAsync(h_grid+h_offset,halo_yz+halo_ind,rsize,cudaMemcpyHostToHost,ctx.d_copy_streams[LEFTRIGHT]);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaHostRegister(grid+offset,mxy*nz*sizeof(real),cudaHostRegisterDefault);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        	cudaMemcpy2DAsync( grid+offset_data, px, d_grid+offset_data, px, sx, ny, cudaMemcpyDeviceToHost, strFront);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaStreamSynchronize(strFront);
src/cuda/src_new/gpu/cuda/core/copyHalosConcur.cu:        cudaHostUnregister(grid+offset);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:#include "concur_cuda_core.cuh"
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:    cudaDeviceGetStreamPriorityRange(&low_prio, &high_prio);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:        cudaStreamCreateWithPriority(&ctx->streams[(StreamName)i], cudaStreamDefault, high_prio + i);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:        cudaEventCreate(&ctx->events[(EventName)i]);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:        cudaStreamDestroy(ctx->streams[(StreamName)i]);
src/cuda/src_new/gpu/cuda/core/concur_cuda_core.cu:        cudaEventDestroy(ctx->events[(EventName)i]);
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:*   Errorhandling for CUDA
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:#define CUDA_ERRCHK_ALWAYS(ans) { cuda_assert((ans), __FILE__, __LINE__); }
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:inline void cuda_assert(cudaError_t code, const char *file, int line, bool abort=true)
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:   if (code != cudaSuccess) {
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:#define CUDA_ERRCHK_KERNEL_ALWAYS() { CUDA_ERRCHK_ALWAYS(cudaPeekAtLastError()); CUDA_ERRCHK_ALWAYS(cudaDeviceSynchronize()); }
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh://host-side concurrency and makes the GPU function calls to execute sequentially
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:    #define CUDA_ERRCHK(ans) { ans; }
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:    #define CUDA_ERRCHK_KERNEL() {}
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:    #define CUDA_ERRCHK(ans) { CUDA_ERRCHK_ALWAYS(ans); }
src/cuda/src_new/gpu/cuda/core/errorhandler_cuda.cuh:    #define CUDA_ERRCHK_KERNEL() { CUDA_ERRCHK(cudaPeekAtLastError()); CUDA_ERRCHK(cudaDeviceSynchronize()); }
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:#include "gpu/cuda/cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void init_grid_cuda_core(Grid* d_grid, Grid* d_grid_dst, CParamConfig* cparams);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void destroy_grid_cuda_core(Grid* d_grid, Grid* d_grid_dst);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void init_halo_cuda_core(GPUContext & ctx, bool firstGPU, bool lastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void destroy_halo_cuda_core(GPUContext & ctx);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void load_grid_cuda_core(Grid* d_grid, CParamConfig* d_cparams, vec3i* h_start_idx, 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void store_grid_cuda_core(Grid* h_grid, CParamConfig* h_cparams, 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void load_forcing_dconsts_cuda_core(ForcingParams* forcing_params);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void update_forcing_coefs_cuda_PC(ForcingParams* fp, CParamConfig* cparams, int start_idx);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void store_slice_cuda_core(Slice* h_slice, CParamConfig* h_cparams, RunConfig* h_run_params, 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:#ifdef GPU_ASTAROTH 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void load_outer_halo_cuda_core(const GPUContext & ctx, Grid* h_grid, real* h_halobuffer, bool lfirstGPU, bool llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void store_internal_halo_cuda_core(const GPUContext & ctx, Grid* h_grid, real* h_halobuffer, bool lfirstGPU, bool llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void load_outer_halo_cuda_core(Grid* d_grid, real* d_halobuffer, CParamConfig* d_cparams,
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void store_internal_halo_cuda_core(Grid* h_grid, real* h_halobuffer, CParamConfig* h_cparams, 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void load_hydro_dconsts_cuda_core(CParamConfig* cparams, RunConfig* run_params, 
src/cuda/src_new/gpu/cuda/core/cuda_core.cuh:void print_gpu_config_cuda_core();
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:#include "cuda_core.cuh"
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:#include "errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void update_forcing_coefs_cuda_PC(ForcingParams* fp, CParamConfig* cparams, int start_idx)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_COEF1, fp->coef1, comSize ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_COEF2, fp->coef2, comSize ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_COEF3, fp->coef3, comSize ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_FDA, fp->fda, comSize ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_FX, fp->fx, comSize*cparams->mx ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_FY, fp->fy, comSize*cparams->my ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol( d_FORCING_FZ, fp->fz+2*start_idx, comSize*cparams->mz ));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void load_forcing_dconsts_cuda_core(ForcingParams* forcing_params)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_KK_VEC_X, &forcing_params->kk_x[k_idx], sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_KK_VEC_Y, &forcing_params->kk_y[k_idx], sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_KK_VEC_Z, &forcing_params->kk_z[k_idx], sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_FORCING_KK_PART_X, &forcing_params->kk_part_x, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_FORCING_KK_PART_Y, &forcing_params->kk_part_y, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_FORCING_KK_PART_Z, &forcing_params->kk_part_z, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_PHI, &forcing_params->phi, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void load_hydro_dconsts_cuda_core(CParamConfig* cparams, RunConfig* run_params, const vec3i start_idx)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nx, &(cparams->nx), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_ny, &(cparams->ny), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nz, &(cparams->nz), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_mx, &(cparams->mx), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_my, &(cparams->my), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_mz, &(cparams->mz), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_mxy, &(cparams->mxy), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nx_min, &(cparams->nx_min), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nx_max, &(cparams->nx_max), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_ny_min, &(cparams->ny_min), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_ny_max, &(cparams->ny_max), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nz_min, &(cparams->nz_min), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_nz_max, &(cparams->nz_max), sizeof(int)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSX, &(cparams->dsx), sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSY, &(cparams->dsy), sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSZ, &(cparams->dsz), sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSX_OFFSET, &dsx_offset, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSY_OFFSET, &dsy_offset, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_DSZ_OFFSET, &dsz_offset, sizeof(real)) ); 
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_XORIG, &xorig, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_YORIG, &yorig, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_ZORIG, &zorig, sizeof(real)) );    
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:	CUDA_ERRCHK( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_NU, &(run_params->nu_visc), sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_CS20, &cs2, sizeof(real)) );	
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaMemcpyToSymbol(d_ETA, &(run_params->eta), sizeof(real)) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void init_grid_cuda_core(Grid* d_grid, Grid* d_grid_dst, CParamConfig* cparams)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMalloc(&(d_grid->arr[i]), grid_size_bytes) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemset(d_grid->arr[i], INT_MAX, grid_size_bytes) );  //MR: What is INT_MAX?
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMalloc(&(d_grid_dst->arr[i]), grid_size_bytes) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemset(d_grid_dst->arr[i], INT_MAX, grid_size_bytes) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void destroy_grid_cuda_core(Grid* d_grid, Grid* d_grid_dst)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaFree(d_grid->arr[i]) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaFree(d_grid_dst->arr[i]) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void load_grid_cuda_core(Grid* d_grid, CParamConfig* d_cparams, vec3i* h_start_idx, Grid* h_grid, CParamConfig* h_cparams)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemcpy(&(d_grid->arr[w][0]), &(h_grid->arr[w][h_start_idx->z*slice_size]), grid_size_bytes, cudaMemcpyHostToDevice) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void store_grid_cuda_core(Grid* h_grid, CParamConfig* h_cparams, Grid* d_grid, CParamConfig* d_cparams, vec3i* h_start_idx)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemcpy(&(h_grid->arr[w][z_offset + h_start_idx->z*slice_size]), &(d_grid->arr[w][z_offset]), grid_size_bytes, cudaMemcpyDeviceToHost) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void store_slice_cuda_core(Slice* h_slice, CParamConfig* h_cparams, RunConfig* h_run_params, Slice* d_slice, CParamConfig* d_cparams, vec3i* h_start_idx)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemcpy(buffer.arr[w], d_slice->arr[w], slice_size_bytes, cudaMemcpyDeviceToHost) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void init_halo_cuda_core(GPUContext & ctx, bool lfirstGPU, bool llastGPU)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    //printf("init_halo_cuda_core\n");
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    initHaloConcur(ctx, lfirstGPU, llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void destroy_halo_cuda_core(GPUContext & ctx)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    //printf("destroy_halo_cuda_core\n");
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    CUDA_ERRCHK( cudaFree(ctx.d_halobuffer) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:      CUDA_ERRCHK(cudaStreamDestroy(ctx.d_copy_streams[i]));
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void load_outer_halo_cuda_core(const GPUContext & ctx, Grid* h_grid, real* h_halobuffer, bool lfirstGPU, bool llastGPU)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    //printf("load_outer_halo_cuda_core\n");
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyOxyPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyOxzPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyOyzPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    //CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    synchronizeStreams(ctx,lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        //!!!unlockHostMemOuter(ctx,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void store_internal_halo_cuda_core(const GPUContext & ctx, Grid* h_grid, real* h_halobuffer, bool lfirstGPU, bool llastGPU)
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    //printf("store_internal_halo_cuda_core\n");
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyIxyPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyIxzPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        copyIyzPlates(ctx,w,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    synchronizeStreams(ctx,lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        //!!!unlockHostMemInner(ctx,h_grid->arr[w],lfirstGPU,llastGPU);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:void print_gpu_config_cuda_core()
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    if (cudaGetDeviceCount(&n_devices) != cudaSuccess) { CRASH("No CUDA devices found!"); }
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    printf("Num CUDA devices found: %u\n", n_devices);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    cudaGetDevice(&initial_device);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        cudaSetDevice(i);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        cudaDeviceProp props;
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        cudaGetDeviceProperties(&props, i);
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        CUDA_ERRCHK( cudaMemGetInfo(&free_bytes, &total_bytes) );
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:        //printf("    Single to double perf. ratio: %dx\n", props.singleToDoublePrecisionPerfRatio); //Not supported with older CUDA versions
src/cuda/src_new/gpu/cuda/core/cuda_core.cu:    cudaSetDevice(initial_device);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:#include <cuda_runtime.h>
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:#include "core/concur_cuda_core.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:#include "generic/collectiveops_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:*       single-GPU implementations on some specific GPU.
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:*       These contexts are stored in a static global array, gpu_contexts.
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:*       device to (1) and pass the necessary information from gpu_contexts[1] to the
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:    CParamConfig    d_cparams;          //Local CParamConfig for the device (GPU-specific grid)
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:    cudaStream_t    d_copy_streams[NUM_COPY_STREAMS];  //Streams for halo copying.
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:} GPUContext;
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void init_cuda_generic(CParamConfig* cparamconf, RunConfig* runconf);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void initialize_cuda_generic(CParamConfig* cparamconf, RunConfig* runconf, const Grid & h_grid);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void destroy_cuda_generic();
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void load_grid_cuda_generic(Grid* h_grid);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void store_grid_cuda_generic(Grid* h_grid);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void load_forcing_params_cuda_generic(ForcingParams* forcing_params);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void update_forcing_coefs_cuda_generic(ForcingParams* forcing_params);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void load_outer_halos_cuda_generic(Grid* g, real* halo);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void store_internal_halos_cuda_generic(Grid* g, real* halo);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void integrate_cuda_generic(real dt);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void integrate_step_cuda_generic(int isubstep, real dt);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void boundcond_step_cuda_generic();
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void exchange_halos_cuda_generic(bool circular);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:real reduce_cuda_PC(ReductType t, GridType a);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:real reduce_cuda_generic(ReductType t, GridType a);
src/cuda/src_new/gpu/cuda/cuda_generic.cuh:void get_slice_cuda_generic(Slice* h_slice);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:*   Implementation for the generic cuda solution.
src/cuda/src_new/gpu/cuda/cuda_generic.cu:*   Manages multiple GPUs on a single node using single-GPU implementations
src/cuda/src_new/gpu/cuda/cuda_generic.cu:*   defined in cuda subdirectories (cuda/core, cuda/generic etc)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "core/cuda_core.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "generic/rk3_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "generic/boundcond_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#include "generic/slice_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:* 	(f.ex. the grid dimensions before it has been decomposed for each GPU)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:static GPUContext* gpu_contexts;
src/cuda/src_new/gpu/cuda/cuda_generic.cu:static inline cudaStream_t get_stream(const int device_id, const StreamName str)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    return gpu_contexts[device_id].concur_ctx.streams[str];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:static inline cudaEvent_t get_event(const int device_id, const EventName ev)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    return gpu_contexts[device_id].concur_ctx.events[ev];    
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    cudaGetDevice(&curr_device);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:	        cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    cudaSetDevice(curr_device);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaDeviceCanAccessPeer(&can_access, device_id, peer);   //MR: information not used
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaDeviceEnablePeerAccess(peer_front, 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaDeviceDisablePeerAccess(peer_front);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaDeviceEnablePeerAccess(peer_back, 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaDeviceDisablePeerAccess(peer_back);    
src/cuda/src_new/gpu/cuda/cuda_generic.cu:*	Handles the allocation and initialization of the memories of all GPUs on
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void init_cuda_generic(CParamConfig* cparamconf, RunConfig* runconf)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (is_initialized) { CRASH("cuda_generic already initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    cudaGetDeviceCount(&num_devices);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    gpu_contexts = (GPUContext*) malloc(sizeof(GPUContext)*num_devices);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    print_gpu_config_cuda_core();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    GPUContext* ctx;
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        //printf("%d\n", __CUDA_ARCH__);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:                "device supports the CUDA architecture you are compiling for.\n"
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        CUDA_ERRCHK_KERNEL_ALWAYS();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void initialize_cuda_generic(CParamConfig* cparamconf, RunConfig* runconf, const Grid & h_grid){
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    GPUContext* ctx;
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        //Allocate and init memory on the GPU
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        init_grid_cuda_core(&ctx->d_grid, &ctx->d_grid_dst, &ctx->d_cparams);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        init_slice_cuda_generic(&ctx->d_slice, &ctx->d_cparams, &h_run_params);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        init_reduction_array_cuda_generic(&ctx->d_reduct_arr, &ctx->d_cparams);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        init_halo_cuda_core(*ctx,device_id==0,device_id==num_devices-1); //Note: Called even without multi-node */
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        load_hydro_dconsts_cuda_core(&ctx->d_cparams, &h_run_params, ctx->start_idx);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:*	Deallocates all memory on the GPU
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void destroy_cuda_generic()
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!"); }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];    
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        destroy_slice_cuda_generic(&ctx->d_slice);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        destroy_reduction_array_cuda_generic(&ctx->d_reduct_arr);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        destroy_grid_cuda_core(&ctx->d_grid, &ctx->d_grid_dst);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        destroy_halo_cuda_core(*ctx);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaDeviceReset();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    free(gpu_contexts);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void load_grid_cuda_generic(Grid* h_grid)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        load_grid_cuda_core(&ctx->d_grid, &ctx->d_cparams, &ctx->start_idx, h_grid, &h_cparams); 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void store_grid_cuda_generic(Grid* h_grid)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        store_grid_cuda_core(h_grid, &h_cparams, &ctx->d_grid, &ctx->d_cparams, &ctx->start_idx); 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:static void local_boundconds_cuda_generic()
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        periodic_xy_boundconds_cuda_generic(&ctx->d_grid, &ctx->d_cparams, 0);    
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        const cudaEvent_t local_bc_done = get_event(device_id, EVENT_LOCAL_BC_DONE);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaEventRecord(local_bc_done, 0);//Implicit synchronization with the default stream        
src/cuda/src_new/gpu/cuda/cuda_generic.cu:static void fetch_halos_cuda_generic(GPUContext* ctx, const int device_id, cudaStream_t stream=0, bool lback=true, bool lfront=true)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:          CUDA_ERRCHK( cudaMemcpyPeerAsync(&ctx->d_grid.arr[w][z_dst0], device_id, 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:                                           &gpu_contexts[back_id].d_grid.arr[w][z_src0], back_id,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:          CUDA_ERRCHK( cudaMemcpyPeerAsync(&ctx->d_grid.arr[w][z_dst1], device_id, 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:                                           &gpu_contexts[front_id].d_grid.arr[w][z_src1], front_id,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void exchange_halos_cuda_generic(bool circular=true)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    GPUContext* ctx;
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    cudaStream_t global_stream;
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:          cudaStreamWaitEvent(global_stream, get_event(peer_front, EVENT_LOCAL_BC_DONE), 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:          cudaStreamWaitEvent(global_stream, get_event(peer_back, EVENT_LOCAL_BC_DONE), 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        fetch_halos_cuda_generic(ctx, device_id, global_stream,circular||device_id>0,circular||device_id<num_devices-1);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void boundcond_step_cuda_generic()
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    local_boundconds_cuda_generic();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    exchange_halos_cuda_generic();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void integrate_step_cuda_generic(int isubstep, real dt)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    //For all GPUs in the node in parallel
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        rk3_inner_cuda_generic(&ctx->d_grid, &ctx->d_grid_dst, isubstep, dt,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        //If fetch_halos_cuda_generic() is not already be scheduled for execution
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        //on the GPU, then the execution order will be wrong
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        rk3_outer_cuda_generic(&ctx->d_grid, &ctx->d_grid_dst, isubstep, dt,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    //code in parallel with the GPU integration/memory transfers
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void integrate_cuda_generic(real dt)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!") }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        boundcond_step_cuda_generic();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        integrate_step_cuda_generic(isubstep, dt);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            periodic_xy_boundconds_cuda_generic(&ctx->d_grid, &ctx->d_cparams, 0);    
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            const cudaEvent_t local_bc_done = get_event(device_id, EVENT_LOCAL_BC_DONE);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaEventRecord(local_bc_done, 0);//Implicit synchronization with the default stream
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            rk3_inner_cuda_generic(&ctx->d_grid, &ctx->d_grid_dst, isubstep, dt, 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            const cudaStream_t global_stream = get_stream(device_id, STREAM_GLOBAL);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaStreamWaitEvent(global_stream, get_event(peer_front, EVENT_LOCAL_BC_DONE), 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaStreamWaitEvent(global_stream, get_event(peer_back, EVENT_LOCAL_BC_DONE), 0);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            fetch_halos_cuda_generic(ctx, device_id, global_stream);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            rk3_outer_cuda_generic(&ctx->d_grid, &ctx->d_grid_dst, isubstep, dt, 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/cuda_generic.cu:real reduce_cuda_PC(ReductType t, GridType grid_type)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            res[device_id] = get_reduction_cuda_generic(&ctx->d_reduct_arr, t, &ctx->d_cparams,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            res[device_id] = get_reduction_cuda_generic(&ctx->d_reduct_arr, t, &ctx->d_cparams, ctx->d_grid.arr[grid_type]);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    //Bruteforce: find max, min or rms from the gpu results
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            CRASH("Unexpected ReductType in reduce_cuda_PC)");
src/cuda/src_new/gpu/cuda/cuda_generic.cu:real reduce_cuda_generic(ReductType t, GridType grid_type)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!"); }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            //    printf("Note: other than NOT_APPLICABLE passed to reduce_cuda_generic as ArrType."
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            res[device_id] = get_reduction_cuda_generic(&ctx->d_reduct_arr, t, &ctx->d_cparams,
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            //if (grid_type == NOT_APPLICABLE) { CRASH("Invalid GridType in reduce_cuda_generic"); }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            res[device_id] = get_reduction_cuda_generic(&ctx->d_reduct_arr, t, &ctx->d_cparams, ctx->d_grid.arr[grid_type]);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    //Bruteforce: find max, min or rms from the gpu results
src/cuda/src_new/gpu/cuda/cuda_generic.cu:            CRASH("Unexpected ReductType in reduce_cuda_generic()");
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void get_slice_cuda_generic(Slice* h_slice)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:    if (!is_initialized) { CRASH("cuda_generic wasn't initialized!"); }
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        update_slice_cuda_generic(&ctx->d_slice, &ctx->d_grid, &ctx->d_cparams, &h_run_params);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        store_slice_cuda_core(h_slice, &h_cparams, &h_run_params, &ctx->d_slice, &ctx->d_cparams, &ctx->start_idx);
src/cuda/src_new/gpu/cuda/cuda_generic.cu://cd src/build/ && make -j && ac_srun_taito_multigpu 4 && cd ../../ && screen py_animate_data --nslices=100
src/cuda/src_new/gpu/cuda/cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void update_forcing_coefs_cuda_generic(ForcingParams* forcing_params){
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        update_forcing_coefs_cuda_PC(forcing_params,&ctx->d_cparams,ctx->start_idx.z);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void load_forcing_params_cuda_generic(ForcingParams* forcing_params)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        GPUContext* ctx = &gpu_contexts[device_id];
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        load_forcing_dconsts_cuda_core(forcing_params);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void load_outer_halos_cuda_generic(Grid* h_grid, real* h_halobuffer)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        load_outer_halo_cuda_core(gpu_contexts[device_id],h_grid, h_halobuffer, device_id==0, device_id==num_devices-1); 
src/cuda/src_new/gpu/cuda/cuda_generic.cu:void store_internal_halos_cuda_generic(Grid* h_grid, real* h_halobuffer)
src/cuda/src_new/gpu/cuda/cuda_generic.cu:      cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:      cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:        store_internal_halo_cuda_core(gpu_contexts[device_id],h_grid, h_halobuffer, device_id==0, device_id==num_devices-1);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:      cudaSetDevice(device_id);
src/cuda/src_new/gpu/cuda/cuda_generic.cu:      cudaDeviceSynchronize();
src/cuda/src_new/gpu/cuda/generic/diff_cuda_generic.cuh:*   Difference formulae used for integration in rk3_cuda_generic.
src/cuda/src_new/gpu/cuda/generic/diff_cuda_generic.cuh:#include "gpu/cuda/core/dconsts_core.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#include "rk3_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#include "diff_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:static void rk3_inner_step_cuda_generic(Grid* d_grid, Grid* d_grid_dst, const real dt, CParamConfig* cparams, cudaStream_t hydro_stream, cudaStream_t induct_stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:static void rk3_outer_step_cuda_generic(Grid* d_grid, Grid* d_grid_dst, const real dt, CParamConfig* cparams, cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:void rk3_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:                      cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:void rk3_inner_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:                      cudaStream_t hydro_stream, cudaStream_t induct_stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_inner_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_inner_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_inner_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:void rk3_outer_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:                      cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_outer_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_outer_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            rk3_outer_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cuh:void init_slice_cuda_generic(Slice* d_slice, CParamConfig* cparamconf, RunConfig* run_params);
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cuh:void destroy_slice_cuda_generic(Slice* d_slice);
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cuh:void update_slice_cuda_generic(Slice* d_slice, Grid* d_grid, CParamConfig* cparams, RunConfig* run_params);
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:#include "slice_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:#include "gpu/cuda/core/dconsts_core.cuh"
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:void init_slice_cuda_generic(Slice* d_slice, CParamConfig* cparams, RunConfig* run_params)
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:        CUDA_ERRCHK( cudaMalloc(&d_slice->arr[i], slice_size) );
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:void destroy_slice_cuda_generic(Slice* d_slice)
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:        CUDA_ERRCHK( cudaFree(d_slice->arr[i]) );
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:__global__ void slice_cuda_generic(Slice & slice, Grid & grid)
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:void update_slice_cuda_generic(Slice* d_slice, Grid* d_grid, CParamConfig* cparams, RunConfig* run_params)
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:    //CUDA call
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:    slice_cuda_generic<'z'><<<blocks_per_grid, threads_per_block>>>(*d_slice, *d_grid);
src/cuda/src_new/gpu/cuda/generic/slice_cuda_generic.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:void rk3_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:                      CParamConfig* cparams, cudaStream_t stream=0);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:void rk3_inner_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:                      CParamConfig* cparams, cudaStream_t hydro_stream=0, cudaStream_t induct_stream=0);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:void rk3_outer_cuda_generic(Grid* d_grid, Grid* d_grid_dst, 
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic.cuh:                      CParamConfig* cparams, cudaStream_t stream=0);
src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cu:#include "gpu/cuda/core/dconsts_core.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cu:                                   const cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cuh:void init_reduction_array_cuda_generic(ReductionArray* reduct_arr, CParamConfig* cparams);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cuh:void destroy_reduction_array_cuda_generic(ReductionArray* reduct_arr);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cuh:real get_reduction_cuda_generic(ReductionArray* reduct_arr, ReductType t, CParamConfig* cparams, 
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:#include "boundcond_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:#include "gpu/cuda/core/dconsts_core.cuh"
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:void boundcond_cuda_generic(Grid* d_grid, CParamConfig* cparams, cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:				CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:				CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:				CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			printf("INVALID X TYPE IN BOUNDCOND_CUDA!\n");
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:				CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			printf("INVALID Y TYPE IN BOUNDCOND_CUDA!\n");
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:			printf("INVALID Z TYPE IN BOUNDCOND_CUDA!\n");
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:void periodic_xy_boundconds_cuda_generic(Grid* d_grid, CParamConfig* cparams, cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:	    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:*   http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:#include "collectiveops_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:#include "gpu/cuda/core/dconsts_core.cuh"
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:void init_reduction_array_cuda_generic(ReductionArray* reduct_arr, CParamConfig* cparams)
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:    CUDA_ERRCHK( cudaMalloc((real**) &reduct_arr->d_vec_res, sizeof(real)) );
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:	CUDA_ERRCHK( cudaMalloc((real**) &reduct_arr->d_partial_result, sizeof(real)*blocks_total) );
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:void destroy_reduction_array_cuda_generic(ReductionArray* reduct_arr)
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:    CUDA_ERRCHK( cudaFree(reduct_arr->d_vec_res) ); reduct_arr->d_vec_res = NULL;
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:    CUDA_ERRCHK( cudaFree(reduct_arr->d_partial_result) ); reduct_arr->d_partial_result = NULL;
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:void reduce_cuda_generic(ReductionArray* reduct_arr, CParamConfig* cparams, real* d_vec_x, real* d_vec_y = NULL, real* d_vec_z = NULL)
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        CRASH("Incorrect BLOCKS_TOTAL in reduce_cuda_generic()")
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:			reduce_initial<512, reduce_op, reduce_init_op><<<bpg, tpb, SMEM_PER_BLOCK>>>(reduct_arr->d_partial_result, d_vec_x, d_vec_y, d_vec_z); CUDA_ERRCHK_KERNEL(); break;
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:			reduce_initial<256, reduce_op, reduce_init_op><<<bpg, tpb, SMEM_PER_BLOCK>>>(reduct_arr->d_partial_result, d_vec_x, d_vec_y, d_vec_z); CUDA_ERRCHK_KERNEL(); break;
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:			reduce_initial<128, reduce_op, reduce_init_op><<<bpg, tpb, SMEM_PER_BLOCK>>>(reduct_arr->d_partial_result, d_vec_x, d_vec_y, d_vec_z); CUDA_ERRCHK_KERNEL(); break;
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        reduce<1024, reduce_op><<<1, 1024, 1024*sizeof(real)>>>(reduct_arr->d_vec_res, reduct_arr->d_partial_result, BLOCKS_TOTAL); CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        reduce<512, reduce_op><<<1, 512, 512*sizeof(real)>>>(reduct_arr->d_vec_res, reduct_arr->d_partial_result, BLOCKS_TOTAL); CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        reduce<256, reduce_op><<<1, 256, 256*sizeof(real)>>>(reduct_arr->d_vec_res, reduct_arr->d_partial_result, BLOCKS_TOTAL); CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        reduce<128, reduce_op><<<1, 128, 128*sizeof(real)>>>(reduct_arr->d_vec_res, reduct_arr->d_partial_result, BLOCKS_TOTAL); CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:        reduce<16, reduce_op><<<1, 16, 16*sizeof(real)>>>(reduct_arr->d_vec_res, reduct_arr->d_partial_result, BLOCKS_TOTAL); CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:real get_reduction_cuda_generic(ReductionArray* reduct_arr, ReductType t, CParamConfig* cparams, real* d_a, real* d_b, real* d_c)
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dmax, ddist>(reduct_arr, cparams, d_a, d_b, d_c);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dmin, ddist>(reduct_arr, cparams, d_a, d_b, d_c);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dsum, dsqrsum>(reduct_arr, cparams, d_a, d_b, d_c);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dmax, dscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dmin, dscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dsum, dsqrscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dsum, dexpsqrscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dsum, dscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:            reduce_cuda_generic<dsum, dexpscal>(reduct_arr, cparams, d_a);
src/cuda/src_new/gpu/cuda/generic/collectiveops_cuda_generic.cu:    CUDA_ERRCHK( cudaMemcpy(&res, (real*)reduct_arr->d_vec_res, sizeof(real), cudaMemcpyDeviceToHost) );
src/cuda/src_new/gpu/cuda/generic/rk3_entropy.cuh:                                   const cudaStream_t stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:#include "rk3_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:#include "diff_cuda_generic.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:#include "gpu/cuda/core/errorhandler_cuda.cuh"
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:static void rk3_step_cuda_generic(Grid* d_grid, Grid* d_grid_dst, const real dt, CParamConfig* cparams, cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:rk3_inner_step_cuda_generic(const Grid* d_grid, Grid* d_grid_dst, const real dt,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                            const cudaStream_t hydro_stream,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                            const cudaStream_t induct_stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:rk3_outer_step_cuda_generic(const Grid* d_grid, Grid* d_grid_dst, const real dt,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                            const cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:    CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:        CUDA_ERRCHK_KERNEL();
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:void rk3_cuda_generic(Grid* d_grid, Grid* d_grid_dst,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                      cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:rk3_inner_cuda_generic(const Grid* d_grid, Grid* d_grid_dst,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                       const cudaStream_t hydro_stream,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                       const cudaStream_t induct_stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_inner_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_inner_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_inner_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, hydro_stream, induct_stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:rk3_outer_cuda_generic(const Grid* d_grid, Grid* d_grid_dst,
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:                       const cudaStream_t stream)
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_outer_step_cuda_generic<0>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_outer_step_cuda_generic<1>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            rk3_outer_step_cuda_generic<2>(d_grid, d_grid_dst, dt, cparams, stream);
src/cuda/src_new/gpu/cuda/generic/rk3_cuda_generic_from_astaroth_master_for_reference_remove_me.cu:            CRASH("Invalid step number in rk3_cuda_generic");
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cuh:void boundcond_cuda_generic(Grid* d_grid, CParamConfig* cparams, cudaStream_t stream=0);
src/cuda/src_new/gpu/cuda/generic/boundcond_cuda_generic.cuh:void periodic_xy_boundconds_cuda_generic(Grid* d_grid, CParamConfig* cparams, cudaStream_t stream=0);
src/cuda/src_new/gpu/gpu.h:*   Interface for accessing the GPU functions
src/cuda/src_new/gpu/gpu.h:#ifdef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.h:    CUDA_GENERIC = 0,
src/cuda/src_new/gpu/gpu.h:    CUDA_19P,
src/cuda/src_new/gpu/gpu.h:    CUDA_55P,
src/cuda/src_new/gpu/gpu.h:    CUDA_MAXWELL,
src/cuda/src_new/gpu/gpu.h:    CUDA_FOR_PENCIL,
src/cuda/src_new/gpu/gpu.h://///Description of the multi-GPU interface///////////////////////////////////////////
src/cuda/src_new/gpu/gpu.h:NOTE: All of the following functions operate on all GPUs on the node unless otherwise stated.
src/cuda/src_new/gpu/gpu.h:GPUInitFunc: 
src/cuda/src_new/gpu/gpu.h:	Starting point of all GPU computation. Handles the allocation and
src/cuda/src_new/gpu/gpu.h:        initialization of *all memory needed on all GPUs in the node*. In other words,
src/cuda/src_new/gpu/gpu.h:        setups everything GPU-side so that calling any other GPU interface function
src/cuda/src_new/gpu/gpu.h:GPUDestroyFunc:
src/cuda/src_new/gpu/gpu.h:	Opposite of GPUInitFunc. Frees all GPU allocations and resets all devices in
src/cuda/src_new/gpu/gpu.h:GPULoadFunc:
src/cuda/src_new/gpu/gpu.h:        into GPU memories.
src/cuda/src_new/gpu/gpu.h:GPUStoreFunc:
src/cuda/src_new/gpu/gpu.h:	Combines and stores the grids in GPU memories into the host grid g.
src/cuda/src_new/gpu/gpu.h:GPULoadForcingParamsFunc:
src/cuda/src_new/gpu/gpu.h:GPULoadOuterHalosFunc:
src/cuda/src_new/gpu/gpu.h:	Similar to GPULoadFunc, but loads only the ghost zone of the host grid into
src/cuda/src_new/gpu/gpu.h:        appropriate locations in GPU memory. TODO review this description.
src/cuda/src_new/gpu/gpu.h:GPUStoreInternalHalosFunc:
src/cuda/src_new/gpu/gpu.h:	Similar to GPUStoreFunc, but combines the ghost zones of the GPU grid and
src/cuda/src_new/gpu/gpu.h:GPUIntegrateFunc:
src/cuda/src_new/gpu/gpu.h:GPUIntegrateStepFunc:
src/cuda/src_new/gpu/gpu.h:GPUBoundcondStepFunc:
src/cuda/src_new/gpu/gpu.h:	Applies the boundary conditions on the grids in GPU memory
src/cuda/src_new/gpu/gpu.h:GPUReduceFunc:
src/cuda/src_new/gpu/gpu.h:	Performs a reduction on the GPU grids. The possible ReductTypes and GridTypes
src/cuda/src_new/gpu/gpu.h:        Usage f.ex. GPUReduce(MAX_SCAL_UU, LNRHO); finds the maximum in the lnrho arrays
src/cuda/src_new/gpu/gpu.h:        on the GPUs.
src/cuda/src_new/gpu/gpu.h:GPUGetSliceFunc:
src/cuda/src_new/gpu/gpu.h:	Similar to GPUStoreFunc, but instead stores a 2D slices to the Slice struct
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUInitFunc)(CParamConfig* cparamconf, RunConfig* runconf);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUInitializeFunc)(CParamConfig* cparamconf, RunConfig* runconf, const Grid & h_grid);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUDestroyFunc)();
src/cuda/src_new/gpu/gpu.h:typedef void (*GPULoadFunc)(Grid* g);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUStoreFunc)(Grid* g);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPULoadForcingParamsFunc)(ForcingParams* fp);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUUpdateForcingCoefsFunc)(ForcingParams* fp);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPULoadOuterHalosFunc)(Grid* g, real* halo);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUStoreInternalHalosFunc)(Grid* g, real* halo);
src/cuda/src_new/gpu/gpu.h://GPU solver functions
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUIntegrateFunc)(real dt);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUIntegrateStepFunc)(int isubstep, real dt);
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUBoundcondStepFunc)();
src/cuda/src_new/gpu/gpu.h:typedef real (*GPUReduceFunc)(ReductType t, GridType a);
src/cuda/src_new/gpu/gpu.h://Misc GPU functions
src/cuda/src_new/gpu/gpu.h:typedef void (*GPUGetSliceFunc)(Slice* s);
src/cuda/src_new/gpu/gpu.h:extern GPUInitFunc               GPUInit;
src/cuda/src_new/gpu/gpu.h:extern GPUInitializeFunc         GPUInitialize;
src/cuda/src_new/gpu/gpu.h:extern GPUDestroyFunc            GPUDestroy;
src/cuda/src_new/gpu/gpu.h:extern GPULoadFunc               GPULoad;             //Load from host to device
src/cuda/src_new/gpu/gpu.h:extern GPUStoreFunc              GPUStore;            //Store from device to host
src/cuda/src_new/gpu/gpu.h:#ifndef GPU_ASTAROTH
src/cuda/src_new/gpu/gpu.h:extern GPULoadForcingParamsFunc  GPULoadForcingParams;
src/cuda/src_new/gpu/gpu.h:extern GPUUpdateForcingCoefsFunc GPUUpdateForcingCoefs;
src/cuda/src_new/gpu/gpu.h:extern GPULoadOuterHalosFunc     GPULoadOuterHalos;
src/cuda/src_new/gpu/gpu.h:extern GPUStoreInternalHalosFunc GPUStoreInternalHalos;
src/cuda/src_new/gpu/gpu.h://GPU solver interface
src/cuda/src_new/gpu/gpu.h:extern GPUIntegrateFunc     GPUIntegrate;
src/cuda/src_new/gpu/gpu.h:extern GPUIntegrateStepFunc GPUIntegrateStep;
src/cuda/src_new/gpu/gpu.h:extern GPUBoundcondStepFunc GPUBoundcondStep;
src/cuda/src_new/gpu/gpu.h:extern GPUReduceFunc        GPUReduce;
src/cuda/src_new/gpu/gpu.h:extern GPUGetSliceFunc  GPUGetSlice;
src/cuda/src_new/gpu/gpu.h:void GPUSelectImplementation(ImplType type);
src/cuda/src_new/README:gpu/CMakeLists.txt
src/cuda/src_new/README:gpu/gpu.cc
src/cuda/src_new/README:gpu/gpu.h
src/cuda/src_new/README:gpu/cuda/cuda_generic.cu
src/cuda/src_new/README:gpu/cuda/cuda_generic.cuh
src/cuda/src_new/README:gpu/cuda/core/cuda_core.cu
src/cuda/src_new/README:gpu/cuda/core/cuda_core.cuh
src/cuda/src_new/README:gpu/cuda/core/dconsts_core.cuh
src/cuda/src_new/README:gpu/cuda/core/errorhandler_cuda.cuh
src/cuda/src_new/README:gpu/cuda/generic/boundcond_cuda_generic.cu
src/cuda/src_new/README:gpu/cuda/generic/boundcond_cuda_generic.cuh
src/cuda/src_new/README:gpu/cuda/generic/collectiveops_cuda_generic.cu
src/cuda/src_new/README:gpu/cuda/generic/collectiveops_cuda_generic.cuh
src/cuda/src_new/README:gpu/cuda/generic/diff_cuda_generic.cuh
src/cuda/src_new/README:gpu/cuda/generic/rk3_cuda_generic.cu
src/cuda/src_new/README:gpu/cuda/generic/rk3_cuda_generic.cuh
src/cuda/src_new/README:gpu/cuda/generic/slice_cuda_generic.cu
src/cuda/src_new/README:gpu/cuda/generic/slice_cuda_generic.cuh
src/cuda/src_new/README:gpu_astaroth_new.cc
src/cuda/src/diagnostics.cu:__global__ void check_grid_for_nan_cuda(float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, int* d_nan_count) 
src/cuda/src/diagnostics.cu:	cudaMalloc((int**) &d_nan_count, sizeof(int)); 
src/cuda/src/diagnostics.cu:	cudaMemcpy((int*) d_nan_count, &nan_count, sizeof(int), cudaMemcpyHostToDevice);
src/cuda/src/diagnostics.cu:        cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	check_grid_for_nan_cuda<<<blocksPerGrid, threadsPerBlock>>>(d_lnrho, d_uu_x, d_uu_y, d_uu_z, d_nan_count);
src/cuda/src/diagnostics.cu:	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	cudaMemcpy(&nan_count, (int*) d_nan_count, sizeof(int), cudaMemcpyDeviceToHost); 
src/cuda/src/diagnostics.cu:	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:    		  float* GPU_lnrho, float* GPU_uu_x, float* GPU_uu_y, float* GPU_uu_z) 
src/cuda/src/diagnostics.cu:				if (	isnan(GPU_lnrho[idx]) || 
src/cuda/src/diagnostics.cu:					isnan(GPU_uu_x[idx]) || 	
src/cuda/src/diagnostics.cu:					isnan(GPU_uu_x[idx]) || 
src/cuda/src/diagnostics.cu:					isnan(GPU_uu_x[idx])) 
src/cuda/src/diagnostics.cu:					printf("GPU result contains nan!\n");
src/cuda/src/diagnostics.cu:				if (lnrho_error < abs(CPU_lnrho[idx]-GPU_lnrho[idx]))
src/cuda/src/diagnostics.cu:					lnrho_error = abs(CPU_lnrho[idx]-GPU_lnrho[idx]);
src/cuda/src/diagnostics.cu:				if (uu_x_error < abs(CPU_uu_x[idx]-GPU_uu_x[idx]))
src/cuda/src/diagnostics.cu:					uu_x_error = abs(CPU_uu_x[idx]-GPU_uu_x[idx]);
src/cuda/src/diagnostics.cu:				if (uu_y_error < abs(CPU_uu_y[idx]-GPU_uu_y[idx]))
src/cuda/src/diagnostics.cu:					uu_y_error = abs(CPU_uu_y[idx]-GPU_uu_y[idx]);
src/cuda/src/diagnostics.cu:				if (uu_z_error < abs(CPU_uu_z[idx]-GPU_uu_z[idx]))
src/cuda/src/diagnostics.cu:					uu_z_error = abs(CPU_uu_z[idx]-GPU_uu_z[idx]);				
src/cuda/src/diagnostics.cu:				error += abs(CPU_lnrho[idx]-GPU_lnrho[idx]);
src/cuda/src/diagnostics.cu:				error += abs(CPU_uu_x[idx]-GPU_uu_x[idx]);
src/cuda/src/diagnostics.cu:				error += abs(CPU_uu_y[idx]-GPU_uu_y[idx]);
src/cuda/src/diagnostics.cu:				error += abs(CPU_uu_z[idx]-GPU_uu_z[idx]);
src/cuda/src/diagnostics.cu:	printf("\n\t\tCPU / GPU: %f, %f\n", CPU_uu_y[CX_BOT + CY_BOT*NX + CZ_BOT*NX*NY], GPU_uu_y[CX_BOT + CY_BOT*NX + CZ_BOT*NX*NY]);
src/cuda/src/diagnostics.cu:* Runs diagnostics by checking that both CPU and GPU versions of 
src/cuda/src/diagnostics.cu:* Note: Requires that the GPU memory is already allocated, constants are
src/cuda/src/diagnostics.cu:	////////////////Add model and GPU functions in these arrays//////////////////////////
src/cuda/src/diagnostics.cu:	VecReductionFunctionDevicePointer GPU_vec_reduction_functions[] = {&max_vec_cuda, &min_vec_cuda};
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );	
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			float GPU_result;
src/cuda/src/diagnostics.cu:			GPU_vec_reduction_functions[j](d_umax, d_partial_result, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			cudaMemcpy(&GPU_result, (float*)d_umax, sizeof(float), cudaMemcpyDeviceToHost); 
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			float error = abs(CPU_result - GPU_result);
src/cuda/src/diagnostics.cu:			const float epsilon = 0.00001f; //Because of the CPU-GPU floating-point differences
src/cuda/src/diagnostics.cu:				printf("\tGPU result: %f\n", GPU_result);
src/cuda/src/diagnostics.cu:	////////////////Add model and GPU functions in these arrays//////////////////////////
src/cuda/src/diagnostics.cu:	ScalReductionFunctionDevicePointer GPU_scal_reduction_functions[] = {&max_scal_cuda};
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:			checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );	
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			float GPU_result;
src/cuda/src/diagnostics.cu:			GPU_scal_reduction_functions[j](d_umax, d_partial_result, d_uu_x);
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			cudaMemcpy(&GPU_result, (float*)d_umax, sizeof(float), cudaMemcpyDeviceToHost); 
src/cuda/src/diagnostics.cu:			cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:			float error = abs(CPU_result - GPU_result);
src/cuda/src/diagnostics.cu:			const float epsilon = 0.00001f; //Because of the CPU-GPU floating-point differences
src/cuda/src/diagnostics.cu:				printf("\tGPU result: %f\n", GPU_result);
src/cuda/src/diagnostics.cu:	cudaMemcpyFromSymbol(&dt, d_DT, sizeof(float));
src/cuda/src/diagnostics.cu:	cudaDeviceSynchronize();	
src/cuda/src/diagnostics.cu:	printf("Checking Rungekutta_2N_cuda with d_DT = %f...\n", dt);
src/cuda/src/diagnostics.cu:	float *GPU_lnrho; //Log density
src/cuda/src/diagnostics.cu:        float *GPU_uu_x, *GPU_uu_y, *GPU_uu_z; //velocities
src/cuda/src/diagnostics.cu:        GPU_lnrho = (float*) malloc(sizeof(float)*GRID_SIZE);
src/cuda/src/diagnostics.cu:        GPU_uu_x  = (float*) malloc(sizeof(float)*GRID_SIZE);
src/cuda/src/diagnostics.cu:        GPU_uu_y  = (float*) malloc(sizeof(float)*GRID_SIZE);
src/cuda/src/diagnostics.cu:        GPU_uu_z  = (float*) malloc(sizeof(float)*GRID_SIZE);
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );	
src/cuda/src/diagnostics.cu:		cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:		//GPU	
src/cuda/src/diagnostics.cu:		rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, 
src/cuda/src/diagnostics.cu:		cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(GPU_lnrho, d_lnrho_dest, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(GPU_uu_x,  d_uu_x_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(GPU_uu_y,  d_uu_y_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:		checkErr( cudaMemcpy(GPU_uu_z,  d_uu_z_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:		cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:					GPU_lnrho, GPU_uu_x, GPU_uu_y, GPU_uu_z);
src/cuda/src/diagnostics.cu:		const float epsilon = 0.00001f; //Cutoff in CPU/GPU floating-point error
src/cuda/src/diagnostics.cu:	//printf("Checking Rungekutta_2N_cuda with Courant timestep...\n");
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );	
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//	//host part of timestep_cuda is correct)
src/cuda/src/diagnostics.cu:	//	dt = timestep_cuda(d_umax, d_partial_result, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)) );
src/cuda/src/diagnostics.cu:	//	//Latest GPU version	
src/cuda/src/diagnostics.cu:	//	rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, 
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(GPU_lnrho, d_lnrho_dest, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(GPU_uu_x,  d_uu_x_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(GPU_uu_y,  d_uu_y_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(GPU_uu_z,  d_uu_z_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );	
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//	//Model GPU version	
src/cuda/src/diagnostics.cu:	//	model_rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, 
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(lnrho, d_lnrho_dest, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(uu_x,  d_uu_x_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(uu_y,  d_uu_y_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	checkErr( cudaMemcpy(uu_z,  d_uu_z_dest,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/diagnostics.cu:	//	cudaDeviceSynchronize();
src/cuda/src/diagnostics.cu:	//				GPU_lnrho, GPU_uu_x, GPU_uu_y, GPU_uu_z);
src/cuda/src/diagnostics.cu:	//	//const float epsilon = 0.00001f; //Cutoff in CPU/GPU floating-point error
src/cuda/src/diagnostics.cu:	//	const float epsilon = 0.01f; //Cutoff in CPU/GPU floating-point error
src/cuda/src/diagnostics.cu:        free(GPU_lnrho);
src/cuda/src/diagnostics.cu:        free(GPU_uu_x);
src/cuda/src/diagnostics.cu:        free(GPU_uu_y);
src/cuda/src/diagnostics.cu:        free(GPU_uu_z);
src/cuda/src/shear_old_BACKUP.cu:	//TODO: MODIFY FOR CUDA
src/cuda/src/slice.cuh:void get_slice_cuda( char slice_axis, float* d_slice_lnrho, float* d_slice_uu, 
src/cuda/src/timestep.cuh:float timestep_cuda(float* d_umax, float* d_partial_result, float* d_uu_x, float* d_uu_y, float* d_uu_z);
src/cuda/src/timestep.cuh:void timeseries_diagnostics_cuda(int step, float dt, double t); 
src/cuda/src/boundcond.cu:void boundcond_cuda(float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z)
src/cuda/src/boundcond.cu:	static cudaStream_t per_x_stream = NULL; 
src/cuda/src/boundcond.cu:		cudaStreamCreate(&per_x_stream);
src/cuda/src/boundcond.cu:	static cudaStream_t per_y_stream = NULL; 
src/cuda/src/boundcond.cu:		cudaStreamCreate(&per_y_stream);
src/cuda/src/boundcond.cu:	static cudaStream_t per_z_stream = NULL; 
src/cuda/src/boundcond.cu:		cudaStreamCreate(&per_z_stream);
src/cuda/src/boundcond.cu:void periodic_boundcond_scal_cuda(float* d_scal) {
src/cuda/src/readme.txt:this because CUDA compiler generates essentially c++ code and we would need 
src/cuda/src/readme.txt:function prototype declarations for CUDA and C code to be compatible with each other)
src/cuda/src/readme.txt:* All CPU helper functions calling kernels should be named *_cuda, such as bouncond_cuda() so that
src/cuda/src/makefile.depend:gpu_astaroth.o: gpu_astaroth.cu ../cparam_c.h ../cdata_c.h ../diagnostics_c.h ../density_c.h ../hydro_c.h ../viscosity_c.h ../eos_c.h ../forcing_c.h ../sub_c.h defines_PC.h dconsts.cuh
src/cuda/src/copy_halos.cu:#include "cuda.h"
src/cuda/src/copy_halos.cu:cudaError_t checkErr(cudaError_t result) {
src/cuda/src/copy_halos.cu:  if (result != cudaSuccess) {
src/cuda/src/copy_halos.cu:    fprintf(stderr, "CUDA Runtime Error: %s %d\n", 
src/cuda/src/copy_halos.cu:            cudaGetErrorString(result), result);
src/cuda/src/copy_halos.cu:    assert(result == cudaSuccess);
src/cuda/src/copy_halos.cu:	cudaError_t err = cudaGetLastError();
src/cuda/src/copy_halos.cu:	if (err != cudaSuccess) 
src/cuda/src/copy_halos.cu:	   //printf("checking kernel error: %s\n", cudaGetErrorString(err));
src/cuda/src/copy_halos.cu:	//cudaGetDevice(&device);
src/cuda/src/copy_halos.cu:	//cudaSetDevice(device); //Not yet enabled
src/cuda/src/copy_halos.cu:	//cudaDeviceReset();
src/cuda/src/copy_halos.cu:	//checkErr(cudaMalloc ((void **) &d_halo, sizeof(float)*halo_size));
src/cuda/src/copy_halos.cu:	//checkErr(cudaMalloc ((void **) &d_lnrho, sizeof(int)*d_lnrho_size));	
src/cuda/src/copy_halos.cu:	//printf("\n Packing done now loading halos to GPU\n");
src/cuda/src/copy_halos.cu:	checkErr(cudaMemcpy(d_halo, halo, sizeof(float)*halo_size  ,cudaMemcpyHostToDevice));
src/cuda/src/copy_halos.cu:	//checkErr(cudaMalloc ((void **) &d_halo, sizeof(int)*halo_size));
src/cuda/src/copy_halos.cu:	//checkErr(cudaMalloc ((void **) &d_lnrho, sizeof(int)*d_lnrho_size));	
src/cuda/src/copy_halos.cu:	//printf("\n loading halos and lnrho to GPU\n");
src/cuda/src/copy_halos.cu:	//cudaMemcpy(d_halo, halo, sizeof(int)*halo_size  ,cudaMemcpyHostToDevice);
src/cuda/src/copy_halos.cu:	//cudaMemcpy(d_lnrho, lnrho, sizeof(int)*d_lnrho_size  ,cudaMemcpyHostToDevice); // for testing purpose
src/cuda/src/copy_halos.cu:	cudaMemcpy(halo, d_halo, sizeof(float)*halo_size  ,cudaMemcpyDeviceToHost);
src/cuda/src/slice.cu:void get_slice_cuda( 	char slice_axis, float* d_slice_lnrho, float* d_slice_uu, 
src/cuda/src/slice.cu:			printf("Invalid slice axis in slice.cu:save_slice_cuda()!\n");
src/cuda/src/integrators.cuh:void rungekutta2N_cuda(	float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, 
src/cuda/src/gpu_astaroth_v2.cu://                             gpu_astaroth.cu
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_NX, &NX, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_NY, &NY, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_NZ, &NZ, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_BOUND_SIZE, &BOUND_SIZE, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_X, &COMP_DOMAIN_SIZE_X, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Y, &COMP_DOMAIN_SIZE_Y, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Z, &COMP_DOMAIN_SIZE_Z, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_NELEMENTS_FLOAT, &nelements_float, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_X, &DOMAIN_SIZE_X, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Y, &DOMAIN_SIZE_Y, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Z, &DOMAIN_SIZE_Z, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Y_OFFSET, &h_w_grid_y_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Z_OFFSET, &h_w_grid_z_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Y_OFFSET, &h_grid_y_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Z_OFFSET, &h_grid_z_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CX_TOP, &cx_top, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CY_TOP, &cy_top, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_TOP, &cz_top, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CX_BOT, &CX_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CY_BOT, &CY_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_BOT, &CZ_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DX, &DX, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DY, &DY, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_XORIG, &XORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_YORIG, &YORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_ZORIG, &ZORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_INTERP_ORDER, &interp_order, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_Q_SHEAR, &Q_SHEAR, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_OMEGA, &OMEGA, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_LFORCING, &LFORCING, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_LSHEAR, &LSHEAR, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaMemcpyToSymbol(d_LCORIOLIS, &lcoriolis, sizeof(int)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA1, &h_ALPHA1, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA2, &h_ALPHA2, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA3, &h_ALPHA3, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_BETA1, &h_BETA1, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_BETA2, &h_BETA2, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_BETA3, &h_BETA3, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_NU_VISC, &NU_VISC, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_CS2_SOUND, &CS2_SOUND, sizeof(float)) );	
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_9, &flt_9, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_45, &flt_45, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_60, &flt_60, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_2, &flt_2, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_27, &flt_27, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_270, &flt_270, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_490, &flt_490, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_180, &flt_180, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(float)) );
src/cuda/src/gpu_astaroth_v2.cu:extern "C" void initializeGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho){ 
src/cuda/src/gpu_astaroth_v2.cu:	cudaGetDevice(&device);
src/cuda/src/gpu_astaroth_v2.cu:	//cudaSetDevice(device); //Not yet enabled
src/cuda/src/gpu_astaroth_v2.cu:	cudaDeviceReset();
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_lnrho, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_x, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_y, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_z, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_w_lnrho, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_w_uu_x, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_w_uu_y, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_w_uu_z, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_lnrho_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_x_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_y_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc(&d_uu_z_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_umax, sizeof(float)) );   
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_umin, sizeof(float)) );   
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_urms, sizeof(float)) );   
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_uxrms, sizeof(float)) );  
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_uyrms, sizeof(float)) );  
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_uzrms, sizeof(float)) );  
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_rhorms, sizeof(float)) ); 
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_partial_result, sizeof(float)) );   
src/cuda/src/gpu_astaroth_v2.cu:	checkErr( cudaMalloc( &d_scaldiag, sizeof(float)) );   
src/cuda/src/gpu_astaroth_v2.cu:	  printf("In gpu_astaroth.cu in initializeGPU Device mem allocated: %f MiB\n", (4*sizeof(float)*GRID_SIZE + 4*sizeof(float)*W_GRID_SIZE)/powf(2,20));
src/cuda/src/gpu_astaroth_v2.cu:	printf("in initializeGPU halo_size = %d\n",halo_size);
src/cuda/src/gpu_astaroth_v2.cu:	checkErr(cudaMalloc((float**)&d_halo, sizeof(float)*halo_size));
src/cuda/src/gpu_astaroth_v2.cu:	printf("Inside initializeGPU in gpu_astaroth_v2 &halo, &d_halo  %p %p pointing to  %p %p\n",&halo,&d_halo, halo, d_halo);
src/cuda/src/gpu_astaroth_v2.cu:	printf("Stop: GPU initialized success inside gpu_astaroth_v2.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:extern "C" void substepGPU(float *uu_x, float *uu_y, float *uu_z, float *lnrho, int isubstep, bool full_inner=false, bool full=false){
src/cuda/src/gpu_astaroth_v2.cu:	//cudaSetDevice(0);
src/cuda/src/gpu_astaroth_v2.cu:	printf("Stop: Now inside substepGPU\n");
src/cuda/src/gpu_astaroth_v2.cu:	//printf("Inside initializeGPU in gpu_astaroth_v2 &halo, &d_halo  %p %p pointing to  %p %p\n",&halo,&d_halo, halo, d_halo);
src/cuda/src/gpu_astaroth_v2.cu:	printf("Inside substepGPU in gpu_astaroth_v2 &halo, &d_halo  %p %p pointing to  %p %p\n",&halo,&d_halo, halo, d_halo);
src/cuda/src/gpu_astaroth_v2.cu:	printf("Stop: Going to copy grid to GPU or copyOuterHalos inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:		printf("Stop: Going to copy grid to GPU inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_lnrho_dest, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_x_dest, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_y_dest, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(d_uu_z_dest, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth_v2.cu:		printf("Stop: Going to call copyouterhalostodevice inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)));
src/cuda/src/gpu_astaroth_v2.cu:       	checkErr( cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)));
src/cuda/src/gpu_astaroth_v2.cu:	//printf("stop3: dt = %f in gpu_astaroth.cu\n", dt);
src/cuda/src/gpu_astaroth_v2.cu:	printf("Calling rungekutta2N_cuda\n");
src/cuda/src/gpu_astaroth_v2.cu:	//cudaSetDevice(0);
src/cuda/src/gpu_astaroth_v2.cu:	rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, d_w_lnrho, d_w_uu_x, d_w_uu_y, d_w_uu_z, d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest, isubstep);
src/cuda/src/gpu_astaroth_v2.cu:	//cudaDeviceSynchronize();
src/cuda/src/gpu_astaroth_v2.cu:	cudaMemcpy(&tmp, &d_uu_x_dest[bound_size + bound_size*mx + bound_size*mx*my], sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth_v2.cu:	cudaDeviceSynchronize();*/
src/cuda/src/gpu_astaroth_v2.cu:		checkErr(cudaMemcpy(output, d_uu_x_dest, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost));
src/cuda/src/gpu_astaroth_v2.cu:		//printf("Stop: starting cudaMemcpyDeviceToHost in gpu_astaroth.cu \n");
src/cuda/src/gpu_astaroth_v2.cu:	//printf("Stop: Starting Copying halos in gpu_astaroth.cu \n");
src/cuda/src/gpu_astaroth_v2.cu:        	checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:		checkErr(cudaMemcpy(output, d_uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost));
src/cuda/src/gpu_astaroth_v2.cu:	//printf("Stop: Finished Copying halos in gpu_astaroth.cu \n");
src/cuda/src/gpu_astaroth_v2.cu:	//printf("Stop: Now going inside timeseries_diagnostics_cuda(it, dt, t) in gpu_astaroth.cu \n");
src/cuda/src/gpu_astaroth_v2.cu:        if (ldiagnos) timeseries_diagnostics_cuda(it, dt, t);
src/cuda/src/gpu_astaroth_v2.cu:	//printf("Stop: Finished executing timeseries_diagnostics_cuda(it, dt, t) in gpu_astaroth.cu \n");
src/cuda/src/gpu_astaroth_v2.cu:extern "C" void finalizeGPU()
src/cuda/src/gpu_astaroth_v2.cu:	//checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:	//checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:	//checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:	//checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth_v2.cu:        //cudaEventDestroy( start );
src/cuda/src/gpu_astaroth_v2.cu:        //cudaEventDestroy( stop );
src/cuda/src/gpu_astaroth_v2.cu:	printf("stop1: inside finalizeGPU in gpy_astaroth_v2.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_lnrho) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_uu_x) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_uu_y) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_uu_z) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_umax) ); checkErr( cudaFree(d_umin) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_urms) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_uxrms) ); checkErr( cudaFree(d_uyrms) ); checkErr( cudaFree(d_uzrms) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_rhorms) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_partial_result) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFree(d_halo) );
src/cuda/src/gpu_astaroth_v2.cu:        /*checkErr( cudaFreeHost(slice_lnrho) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFreeHost(slice_uu) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFreeHost(slice_uu_x) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFreeHost(slice_uu_y) );
src/cuda/src/gpu_astaroth_v2.cu:        checkErr( cudaFreeHost(slice_uu_z) );*/
src/cuda/src/gpu_astaroth_v2.cu:	printf("stop2: inside finalizeGPU in gpy_astaroth_v2.cu\n");
src/cuda/src/gpu_astaroth_v2.cu:	cudaDeviceSynchronize();
src/cuda/src/gpu_astaroth_v2.cu:        //checkErr(cudaDeviceReset());
src/cuda/src/gpu_astaroth_v2.cu:	printf("GPU finalized %d", iproc);
src/cuda/src/vuori_template.sh:#SBATCH -p gpu
src/cuda/src/vuori_template.sh:#SBATCH --partition=gpu
src/cuda/src/vuori_template.sh:#SBATCH --gres=gpu:1
src/cuda/src/gpu_astaroth.cuh:void rungekutta2N_cuda(	float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, 
src/cuda/src/TODO.txt:	developer.download.nvidia.com/CUDA/CUDA_Zone/papers/gpu_3dfd_rev.pdf,
src/cuda/src/diagnostics.cuh://Checks CUDA errors; replaced with nop if DEBUG not defined
src/cuda/src/diagnostics.cuh:cudaError_t checkErr(cudaError_t result) {
src/cuda/src/diagnostics.cuh:  if (result != cudaSuccess) {
src/cuda/src/diagnostics.cuh:    fprintf(stderr, "CUDA Runtime Error: %s \n", 
src/cuda/src/diagnostics.cuh:            cudaGetErrorString(result));
src/cuda/src/diagnostics.cuh:    assert(result == cudaSuccess);
src/cuda/src/diagnostics.cuh:	checkErr( cudaPeekAtLastError() );
src/cuda/src/diagnostics.cuh:	checkErr( cudaDeviceSynchronize() );	
src/cuda/src/diagnostics.cuh:__global__ void check_grid_for_nan_cuda(float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, int* d_nan_count);
src/cuda/src/defines.h://Minimum grid spacing (DSMIN used in timestep_cuda) cannot be defined
src/cuda/src/copyouterhalostodevice.cu:   Copying Outer halos from host to GPU
src/cuda/src/copyouterhalostodevice.cu:	/*static cudaStream_t per_row_stream = NULL; 
src/cuda/src/copyouterhalostodevice.cu:		cudaStreamCreate(&per_row_stream);
src/cuda/src/copyouterhalostodevice.cu:	static cudaStream_t per_col_stream = NULL; 
src/cuda/src/copyouterhalostodevice.cu:		cudaStreamCreate(&per_col_stream);
src/cuda/src/copyouterhalostodevice.cu:	static cudaStream_t per_frtbk_stream = NULL; 
src/cuda/src/copyouterhalostodevice.cu:		cudaStreamCreate(&per_frtbk_stream);*/
src/cuda/src/copyouterhalostodevice.cu:	cudaDeviceSynchronize();
src/cuda/src/copyouterhalostodevice.cu:	cudaDeviceSynchronize();
src/cuda/src/copyouterhalostodevice.cu:	cudaDeviceSynchronize();// needs to be commented out at all places after first verification of code
src/cuda/src/copyouterhalostodevice.cu:	cudaDeviceSynchronize();
src/cuda/src/copyouterhalostodevice.cu:	cudaDeviceSynchronize();
src/cuda/src/copyouterhalostodevice.cu:	//checkErr(cudaStreamDestroy(per_row_stream));
src/cuda/src/copyouterhalostodevice.cu:	//checkErr(cudaStreamDestroy(per_col_stream));
src/cuda/src/copyouterhalostodevice.cu:	//checkErr(cudaStreamDestroy(per_frtbk_stream));
src/cuda/src/makefile:deviceobjs = diagnostics.o integrators_v5.o collectiveops.o copyHalosConcur.o timestep.o gpu_astaroth.o copyouterhalostodevice.o copyinternalhalostohost.o copy_halos.o
src/cuda/src/makefile:devicecus = slice.cu collectiveops.cu diagnostics.cu integrators_v5.cu boundcond.cu timestep.cu diff.cu shear.cu coriolis.cu forcing.cu continuity.cu hydro.cu gpu_astaroth.cu copyouterhalostodevice.cu copyinternalhalostohost.cu copy_halos.cu slice.cu
src/cuda/src/makefile:#CUDA thread; Basically low register count increases
src/cuda/src/makefile:# Settings for taito-gpu
src/cuda/src/alcyone_template_memcheck.sh:#SBATCH --partition="2G_gpu3,2G_gpu6"
src/cuda/src/alcyone_template_memcheck.sh:#SBATCH --gres=gpu:1
src/cuda/src/alcyone_template_memcheck.sh:module load cuda/5.5.22
src/cuda/src/alcyone_template_memcheck.sh:cuda-memcheck ./runme 
src/cuda/src/copyHalosConcur.cu:static cudaStream_t strFront=NULL, strBack=NULL, strBot=NULL, strTop=NULL, strLeftRight=NULL;
src/cuda/src/copyHalosConcur.cu:        cudaStreamCreate(&strFront);
src/cuda/src/copyHalosConcur.cu:        cudaStreamCreate(&strBack);
src/cuda/src/copyHalosConcur.cu:        cudaStreamCreate(&strBot);
src/cuda/src/copyHalosConcur.cu:        cudaStreamCreate(&strTop);
src/cuda/src/copyHalosConcur.cu:        cudaStreamCreate(&strLeftRight);
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMalloc((void**) &d_halo_widths_x, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMalloc((void**) &d_halo_widths_y, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMalloc((void**) &d_halo_widths_z, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        checkErr( cudaMemcpyToSymbol(d_halo_widths_x, halo_widths_x, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        checkErr( cudaMemcpyToSymbol(d_halo_widths_y, halo_widths_y, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        checkErr( cudaMemcpyToSymbol(d_halo_widths_z, halo_widths_z, 3*sizeof(int)) );
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMemcpy( d_halo_widths_x, halo_widths_x, 3*sizeof(int), cudaMemcpyHostToDevice ));
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMemcpy( d_halo_widths_y, halo_widths_y, 3*sizeof(int), cudaMemcpyHostToDevice ));
src/cuda/src/copyHalosConcur.cu:        //checkErr( cudaMemcpy( d_halo_widths_z, halo_widths_z, 3*sizeof(int), cudaMemcpyHostToDevice ));
src/cuda/src/copyHalosConcur.cu:        cudaMalloc(&d_halo_yz,halo_yz_size);            // buffer for yz halos in device
src/cuda/src/copyHalosConcur.cu:        cudaFree(&d_halo_yz);
src/cuda/src/copyHalosConcur.cu:        cudaStreamDestroy(strFront);
src/cuda/src/copyHalosConcur.cu:        cudaStreamDestroy(strBack);
src/cuda/src/copyHalosConcur.cu:        cudaStreamDestroy(strBot);
src/cuda/src/copyHalosConcur.cu:        cudaStreamDestroy(strTop);
src/cuda/src/copyHalosConcur.cu:        cudaStreamDestroy(strLeftRight);
src/cuda/src/copyHalosConcur.cu:        cudaHostRegister(grid, size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:        cudaMemcpyAsync(d_grid, grid, size, cudaMemcpyHostToDevice, strFront);
src/cuda/src/copyHalosConcur.cu:        cudaHostRegister(grid+offset, size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:        cudaMemcpyAsync(d_grid+offset, grid+offset, size, cudaMemcpyHostToDevice, strBack);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpyAsync(d_grid+offset, grid+offset, size, cudaMemcpyHostToDevice, strBot);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpyAsync(d_grid+offset, grid+offset, size, cudaMemcpyHostToDevice, strTop);
src/cuda/src/copyHalosConcur.cu:                        cudaMemcpy(halo_yz+halo_ind,grid+offset,halo_widths_x[0]*sizeof(float),cudaMemcpyHostToHost);  // also async?
src/cuda/src/copyHalosConcur.cu:                        cudaMemcpy(halo_yz+halo_ind,grid+offset,halo_widths_x[1]*sizeof(float),cudaMemcpyHostToHost);  // also async?
src/cuda/src/copyHalosConcur.cu:        cudaHostRegister(halo_yz, halo_yz_size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:        cudaMemcpyAsync(d_halo_yz, halo_yz, halo_yz_size, cudaMemcpyHostToDevice, strLeftRight);
src/cuda/src/copyHalosConcur.cu://  unpacking in global memory; done by GPU kernel in stream strLeftRight
src/cuda/src/copyHalosConcur.cu:        cudaDeviceSynchronize();
src/cuda/src/copyHalosConcur.cu:cudaMemcpy(&buf,d_grid+offset,3*sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strFront);
src/cuda/src/copyHalosConcur.cu:	cudaHostUnregister(grid);	
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strBack);
src/cuda/src/copyHalosConcur.cu:	cudaHostUnregister(grid+mxy*(mz-nghost));
src/cuda/src/copyHalosConcur.cu:	cudaStreamSynchronize(strBot);
src/cuda/src/copyHalosConcur.cu:        	cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:	cudaStreamSynchronize(strTop);
src/cuda/src/copyHalosConcur.cu:        	cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strLeftRight);
src/cuda/src/copyHalosConcur.cu:        cudaHostUnregister(halo_yz);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strFront);
src/cuda/src/copyHalosConcur.cu:          cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strBack);
src/cuda/src/copyHalosConcur.cu:          cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strBot);
src/cuda/src/copyHalosConcur.cu:          cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strTop);
src/cuda/src/copyHalosConcur.cu:          cudaHostUnregister(grid+offset);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strLeftRight);
src/cuda/src/copyHalosConcur.cu:        cudaHostUnregister(halo_yz);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, px*ny, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpy2DAsync(grid+offset, px, d_grid+offset, px, sx, ny, cudaMemcpyDeviceToHost, strFront);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, px*ny, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpy2DAsync(grid+offset, px, d_grid+offset, px, sx, ny, cudaMemcpyDeviceToHost, strBack);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, px*halo_widths_y[bot], cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpy2DAsync( grid+offset, px, d_grid+offset, px, sx, halo_widths_y[bot], cudaMemcpyDeviceToHost, strBot);
src/cuda/src/copyHalosConcur.cu:          cudaHostRegister(grid+offset, px*halo_widths_y[top], cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:          cudaMemcpy2DAsync( grid+offset, px, d_grid+offset, px, sx, halo_widths_y[top], cudaMemcpyDeviceToHost, strTop);
src/cuda/src/copyHalosConcur.cu:        cudaHostRegister(halo_yz, halo_size, cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:        cudaMemcpyAsync(halo_yz, d_halo_yz, halo_size, cudaMemcpyDeviceToHost,strLeftRight);
src/cuda/src/copyHalosConcur.cu:                        cudaMemcpyAsync(grid+offset,halo_yz+halo_ind,size_left,cudaMemcpyHostToHost,strLeftRight);
src/cuda/src/copyHalosConcur.cu:                        cudaMemcpyAsync(grid+offset,halo_yz+halo_ind,size_right,cudaMemcpyHostToHost,strLeftRight);
src/cuda/src/copyHalosConcur.cu:	cudaHostRegister(grid,size,cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:	cudaMemcpy(d_grid, grid, size, cudaMemcpyHostToDevice);
src/cuda/src/copyHalosConcur.cu:	cudaHostUnregister(grid);
src/cuda/src/copyHalosConcur.cu:        cudaHostRegister(grid+offset,mxy*nz*sizeof(float),cudaHostRegisterDefault);
src/cuda/src/copyHalosConcur.cu:        	cudaMemcpy2DAsync( grid+offset_data, px, d_grid+offset_data, px, sx, ny, cudaMemcpyDeviceToHost, strFront);
src/cuda/src/copyHalosConcur.cu:        cudaStreamSynchronize(strFront);
src/cuda/src/copyHalosConcur.cu:        cudaHostUnregister(grid+offset);
src/cuda/src/integrators_v5.cu:void rungekutta2N_cuda(	float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, 
src/cuda/src/integrators_v5.cu:	cudaDeviceSynchronize();
src/cuda/src/boundcond.cuh:void boundcond_cuda(float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z);
src/cuda/src/boundcond.cuh:void periodic_boundcond_scal_cuda(float* d_scal);
src/cuda/src/makefile (copy):#CUDA thread; Basically low register count increases
src/cuda/src/compute.cu://CUDA libraries
src/cuda/src/compute.cu://checkErr( cudaMemcpyToSymbol(d_NX, &nx, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_NX, &nx, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_NY, &ny, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_NZ, &nz, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_PAD_SIZE, &pad_size, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_BOUND_SIZE, &bound_size, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_X, &comp_domain_size_x, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Y, &comp_domain_size_y, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Z, &comp_domain_size_z, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_NELEMENTS_FLOAT, &nelements_float, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_X, &domain_size_x, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Y, &domain_size_y, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Z, &domain_size_z, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Y_OFFSET, &h_w_grid_y_offset, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Z_OFFSET, &h_w_grid_z_offset, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Y_OFFSET, &h_grid_y_offset, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Z_OFFSET, &h_grid_z_offset, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CX_TOP, &cx_top, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CY_TOP, &cy_top, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_TOP, &cz_top, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CX_BOT, &cx_bot, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CY_BOT, &cy_bot, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_BOT, &cz_bot, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DX, &dx, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DY, &dy, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DZ, &dz, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_XORIG, &xorig, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_YORIG, &yorig, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_ZORIG, &zorig, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_INTERP_ORDER, &interp_order, sizeof(int)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_Q_SHEAR, &q_shear, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_OMEGA, &omega, sizeof(float)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_LFORCING, &lforcing, sizeof(int)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_LSHEAR, &lshear, sizeof(int)) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_LCORIOLIS, &lcoriolis, sizeof(int)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA1, &h_ALPHA1, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA2, &h_ALPHA2, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA3, &h_ALPHA3, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_BETA1, &h_BETA1, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_BETA2, &h_BETA2, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_BETA3, &h_BETA3, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_NU_VISC, &nu_visc, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_CS2_SOUND, &cs2_sound, sizeof(float)) );	
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_9, &flt_9, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_45, &flt_45, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_60, &flt_60, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_2, &flt_2, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_27, &flt_27, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_270, &flt_270, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_490, &flt_490, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_180, &flt_180, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(float)) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(float)) );
src/cuda/src/compute.cu:	//http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
src/cuda/src/compute.cu:	cudaGetDevice(&device);
src/cuda/src/compute.cu:	//cudaSetDevice(device); //Not yet enabled
src/cuda/src/compute.cu:	cudaDeviceReset();
src/cuda/src/compute.cu:	checkErr( cudaHostAlloc((float**)&slice_lnrho, sizeof(float)*slice_size, cudaHostAllocMapped) );
src/cuda/src/compute.cu:	checkErr( cudaHostAlloc((float**)&slice_uu, sizeof(float)*slice_size, cudaHostAllocMapped) );
src/cuda/src/compute.cu:	checkErr( cudaHostAlloc((float**)&slice_uu_x, sizeof(float)*slice_size, cudaHostAllocMapped) );
src/cuda/src/compute.cu:	checkErr( cudaHostAlloc((float**)&slice_uu_y, sizeof(float)*slice_size, cudaHostAllocMapped) );
src/cuda/src/compute.cu:	checkErr( cudaHostAlloc((float**)&slice_uu_z, sizeof(float)*slice_size, cudaHostAllocMapped) );
src/cuda/src/compute.cu:	checkErr( cudaHostGetDevicePointer((float **)&d_slice_lnrho, (float *)slice_lnrho, 0) );
src/cuda/src/compute.cu:	checkErr( cudaHostGetDevicePointer((float **)&d_slice_uu, (float *)slice_uu, 0) );
src/cuda/src/compute.cu:	checkErr( cudaHostGetDevicePointer((float **)&d_slice_uu_x, (float *)slice_uu_x, 0) );
src/cuda/src/compute.cu:	checkErr( cudaHostGetDevicePointer((float **)&d_slice_uu_y, (float *)slice_uu_y, 0) );
src/cuda/src/compute.cu:	checkErr( cudaHostGetDevicePointer((float **)&d_slice_uu_z, (float *)slice_uu_z, 0) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_lnrho, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_x, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_y, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_z, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_w_lnrho, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_w_uu_x, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_w_uu_y, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_w_uu_z, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_lnrho_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_x_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_y_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_uu_z_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/compute.cu:	checkErr( cudaMalloc(&d_div_uu, sizeof(float)*GRID_SIZE) );	
src/cuda/src/compute.cu:	checkErr( cudaMalloc((float**) &d_umax, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_umin, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_urms, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uxrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uyrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uzrms, sizeof(float)) );  //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_rhorms, sizeof(float)) ); //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_rhomax, sizeof(float)) ); //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_rhomin, sizeof(float)) ); //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uxmax, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uxmin, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uymax, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uymin, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uzmax, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:        checkErr( cudaMalloc((float**) &d_uzmin, sizeof(float)) );   //TODO this somewhere else
src/cuda/src/compute.cu:	//Allocate memory for the partial result used for calculating max velocity in timestep_cuda
src/cuda/src/compute.cu:	checkErr( cudaMalloc((float**) &d_partial_result, sizeof(float)*BLOCKS_TOTAL) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpy(d_lnrho_dest, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpy(d_uu_x_dest, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpy(d_uu_y_dest, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:        checkErr( cudaMemcpy(d_uu_z_dest, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/compute.cu:	get_slice_cuda('z', d_slice_lnrho, d_slice_uu, d_slice_uu_x, d_slice_uu_y, d_slice_uu_z, d_lnrho, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/compute.cu:	cudaThreadSynchronize();
src/cuda/src/compute.cu:        checkErr( cudaMemcpyToSymbol(d_DELTA_Y, &delta_y, sizeof(float)) );
src/cuda/src/compute.cu:	boundcond_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/compute.cu:	boundcond_cuda(d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest);
src/cuda/src/compute.cu:	get_slice_cuda('z', d_slice_lnrho, d_slice_uu, d_slice_uu_x, d_slice_uu_y, d_slice_uu_z, d_lnrho, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/compute.cu:	cudaThreadSynchronize();
src/cuda/src/compute.cu:	checkErr( cudaMemcpyToSymbol(d_FTRIGGER, &ftrigger, sizeof(int)) );
src/cuda/src/compute.cu:	cudaEvent_t start, stop;
src/cuda/src/compute.cu:	cudaEventCreate(&start);
src/cuda/src/compute.cu:	cudaEventCreate(&stop);
src/cuda/src/compute.cu:	dt = timestep_cuda(d_umax, d_partial_result, d_uu_x, d_uu_y, d_uu_z); //// omer remove this line after debug
src/cuda/src/compute.cu:		cudaEventRecord( start, 0 );
src/cuda/src/compute.cu:		dt = timestep_cuda(d_umax, d_partial_result, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/compute.cu:		checkErr( cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_DELTA_Y, &delta_y, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_FTRIGGER, &ftrigger, sizeof(int)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_KK_VEC_X, &kk_vec_x, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_KK_VEC_Y, &kk_vec_y, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_KK_VEC_Z, &kk_vec_z, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_FORCING_KK_PART_X, &forcing_kk_part_x, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_FORCING_KK_PART_Y, &forcing_kk_part_y, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_FORCING_KK_PART_Z, &forcing_kk_part_z, sizeof(float)) );
src/cuda/src/compute.cu:			checkErr( cudaMemcpyToSymbol(d_PHI, &phi, sizeof(float)) );
src/cuda/src/compute.cu:		rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, 
src/cuda/src/compute.cu:		cudaEventRecord( stop, 0 );
src/cuda/src/compute.cu:		cudaEventSynchronize( stop );
src/cuda/src/compute.cu:		cudaEventElapsedTime( &time_elapsed, start, stop );
src/cuda/src/compute.cu:			get_slice_cuda('z', d_slice_lnrho, d_slice_uu, d_slice_uu_x, d_slice_uu_y, d_slice_uu_z, d_lnrho, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/compute.cu:			cudaThreadSynchronize();
src/cuda/src/compute.cu:			timeseries_diagnostics_cuda(d_umax, d_umin, d_urms, d_uxrms, d_uyrms, d_uzrms, d_rhorms, 
src/cuda/src/compute.cu:		//Point is, that kernels are queued on the GPU and CPU stops 
src/cuda/src/compute.cu:		//GPU BOTTLENECK VERSION:
src/cuda/src/compute.cu:		MV: The gpu bottleneck version seems somehow more intuitive for me. 
src/cuda/src/compute.cu:		JP: This GPU bottleneck version should be more efficient with smaller grids
src/cuda/src/compute.cu:		as all GPU computation can focus solely on RK; with bigger grids CPU probably calculates the diagnostics
src/cuda/src/compute.cu:		slower than the GPU calculates the RK, so we would have to transfer some of the computation to
src/cuda/src/compute.cu:		the GPU => GPU will have to sacrifice RK time for calculating diagnostics. 
src/cuda/src/compute.cu:		MV: with CUDA, which we wanted to avoid. Need to think about this... 
src/cuda/src/compute.cu:		already have an ultra optimized GPU code ready for it, and we need it
src/cuda/src/compute.cu:		for RK anyways, so the faster it's done, the better (GPU reductions are 
src/cuda/src/compute.cu:		* 2. Call timestep kernel (GPU)
src/cuda/src/compute.cu:		* 3. Call RK3 kernel (GPU)
src/cuda/src/compute.cu:		* 1. Call timestep kernel (GPU)
src/cuda/src/compute.cu:		* 2. Call RK3 kernel (GPU)
src/cuda/src/compute.cu:		* 3. Calculate slices & diagnostics (GPU)
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/compute.cu:	checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/compute.cu:	/*checkErr( cudaMemcpy(uu_x,  d_w_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/compute.cu:	cudaEventDestroy( start );
src/cuda/src/compute.cu:	cudaEventDestroy( stop );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_lnrho) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_uu_x) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_uu_y) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_uu_z) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_umax) ); checkErr( cudaFree(d_umin) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_urms) );
src/cuda/src/compute.cu: 	checkErr( cudaFree(d_uxrms) ); checkErr( cudaFree(d_uyrms) ); checkErr( cudaFree(d_uzrms) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_rhorms) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_rhomax) ); checkErr( cudaFree(d_rhomin) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_uxmax) ); checkErr( cudaFree(d_uymax) ); checkErr( cudaFree(d_uzmax) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_uxmin) ); checkErr( cudaFree(d_uymin) ); checkErr( cudaFree(d_uzmin) );
src/cuda/src/compute.cu:	checkErr( cudaFree(d_partial_result) );
src/cuda/src/compute.cu:	checkErr( cudaFreeHost(slice_lnrho) );
src/cuda/src/compute.cu:	checkErr( cudaFreeHost(slice_uu) );
src/cuda/src/compute.cu:	checkErr( cudaFreeHost(slice_uu_x) );
src/cuda/src/compute.cu:	checkErr( cudaFreeHost(slice_uu_y) );
src/cuda/src/compute.cu:	checkErr( cudaFreeHost(slice_uu_z) );
src/cuda/src/compute.cu:	cudaDeviceReset();
src/cuda/src/smem.cuh://-----Defines for rungekutta2N_cuda-------------
src/cuda/src/smem.cuh://Dimensions of the thread block used in rungekutta2N_cuda
src/cuda/src/integrators_v5s.cu:void rungekutta2N_cuda(	float* d_lnrho, float* d_uu_x, float* d_uu_y, float* d_uu_z, 
src/cuda/src/integrators_v5s.cu:      //cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	////periodic_boundcond_scal_cuda(d_div_uu); //Boundary conditions for the divergence field 
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	boundcond_cuda(d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest);
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:        //cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	////periodic_boundcond_scal_cuda(d_div_uu); //Boundary conditions for the divergence field 
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	boundcond_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	cudaEvent_t start, stop;
src/cuda/src/integrators_v5s.cu:	cudaEventCreate(&start);
src/cuda/src/integrators_v5s.cu:	cudaEventCreate(&stop);
src/cuda/src/integrators_v5s.cu:	cudaEventRecord( start, 0 );
src/cuda/src/integrators_v5s.cu:        //cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	////periodic_boundcond_scal_cuda(d_div_uu);; //Boundary conditions for the divergence field 
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	boundcond_cuda(d_lnrho_dest, d_uu_x_dest, d_uu_y_dest, d_uu_z_dest);
src/cuda/src/integrators_v5s.cu:	//cudaDeviceSynchronize();
src/cuda/src/integrators_v5s.cu:	cudaEventRecord( stop, 0 );
src/cuda/src/integrators_v5s.cu:	cudaEventSynchronize( stop );
src/cuda/src/integrators_v5s.cu:	cudaEventElapsedTime( &time, start, stop );
src/cuda/src/integrators_v5s.cu:	cudaEventDestroy( start );
src/cuda/src/integrators_v5s.cu:	cudaEventDestroy( stop );
src/cuda/src/integrators_v5s.cu:	cudaDeviceSynchronize();
src/cuda/src/testmain.c:#include "gpu_astaroth.cuh"
src/cuda/src/testmain.c:	finalizeGpu(uu_x, uu_y, uu_z, lnrho);
src/cuda/src/alcyone_template.sh:#SBATCH --partition="2G_gpu3,2G_gpu6"
src/cuda/src/alcyone_template.sh:#SBATCH --gres=gpu:1
src/cuda/src/alcyone_template.sh:module load cuda/5.5.22
src/cuda/src/gpu_astaroth.cu://                             gpu_astaroth.cu
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NX, &NX, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NY, &NY, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NZ, &NZ, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BOUND_SIZE, &BOUND_SIZE, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_X, &COMP_DOMAIN_SIZE_X, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Y, &COMP_DOMAIN_SIZE_Y, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_COMP_DOMAIN_SIZE_Z, &COMP_DOMAIN_SIZE_Z, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NELEMENTS_FLOAT, &nelements_float, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_X, &DOMAIN_SIZE_X, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Y, &DOMAIN_SIZE_Y, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DOMAIN_SIZE_Z, &DOMAIN_SIZE_Z, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Y_OFFSET, &h_w_grid_y_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_W_GRID_Z_OFFSET, &h_w_grid_z_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Y_OFFSET, &h_grid_y_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_GRID_Z_OFFSET, &h_grid_z_offset, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_TOP, &cx_top, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_TOP, &cy_top, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_TOP, &cz_top, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CX_BOT, &CX_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CY_BOT, &CY_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CZ_BOT, &CZ_BOT, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DX, &DX, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DY, &DY, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DZ, &DZ, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_XORIG, &XORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_YORIG, &YORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_ZORIG, &ZORIG, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_INTERP_ORDER, &interp_order, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_Q_SHEAR, &Q_SHEAR, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_OMEGA, &OMEGA, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LFORCING, &LFORCING, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LSHEAR, &LSHEAR, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaMemcpyToSymbol(d_LCORIOLIS, &lcoriolis, sizeof(int)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA1, &h_ALPHA1, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA2, &h_ALPHA2, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_ALPHA3, &h_ALPHA3, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA1, &h_BETA1, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA2, &h_BETA2, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_BETA3, &h_BETA3, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_NU_VISC, &NU_VISC, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_CS2_SOUND, &CS2_SOUND, sizeof(float)) );	
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_9, &flt_9, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_45, &flt_45, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_60, &flt_60, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_2, &flt_2, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_27, &flt_27, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_270, &flt_270, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_490, &flt_490, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_FLT_180, &flt_180, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DX_DIV, &diff1_dx, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DY_DIV, &diff1_dy, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF1_DZ_DIV, &diff1_dz, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DX_DIV, &diff2_dx, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DY_DIV, &diff2_dy, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFF2_DZ_DIV, &diff2_dz, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDY_DIV, &diffmn_dxdy, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DYDZ_DIV, &diffmn_dydz, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMemcpyToSymbol(d_DIFFMN_DXDZ_DIV, &diffmn_dxdz, sizeof(float)) );
src/cuda/src/gpu_astaroth.cu:extern "C" void initGPU(){
src/cuda/src/gpu_astaroth.cu:        cudaGetDevice(&device);
src/cuda/src/gpu_astaroth.cu:        //cudaSetDevice(device); //Not yet enabled
src/cuda/src/gpu_astaroth.cu:        cudaDeviceReset();
src/cuda/src/gpu_astaroth.cu:extern "C" void registerGPU(float* f){ 
src/cuda/src/gpu_astaroth.cu:extern "C" void initializeGPU(){ 
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_lnrho, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_x, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_y, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_z, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_w_lnrho, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_w_uu_x, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_w_uu_y, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_w_uu_z, sizeof(float)*W_GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_lnrho_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_x_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_y_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc(&d_uu_z_dest, sizeof(float)*GRID_SIZE) );
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc( &d_partial_result, sizeof(float)) );   
src/cuda/src/gpu_astaroth.cu:	checkErr( cudaMalloc( &d_scaldiag, sizeof(float)) );   
src/cuda/src/gpu_astaroth.cu:        	printf("in initializeGPU halo_size = %d\n",halo_size);
src/cuda/src/gpu_astaroth.cu:        	checkErr(cudaMalloc((float**)&d_halo, sizeof(float)*halo_size));
src/cuda/src/gpu_astaroth.cu:        	printf("Inside initializeGPU in gpu_astaroth &halo, &d_halo  %p %p pointing to  %p %p\n",&halo,&d_halo, halo, d_halo);
src/cuda/src/gpu_astaroth.cu:extern "C" void substepGPU(int isubstep, bool full=false){
src/cuda/src/gpu_astaroth.cu:        	printf("Stop: Going to copy grid to GPU or copyOuterHalos inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth.cu:                	printf("Stop: Going to copy grid to GPU inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_lnrho, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_x, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_y, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_z, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_lnrho_dest, lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_x_dest, uu_x, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_y_dest, uu_y, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:                	checkErr( cudaMemcpy(d_uu_z_dest, uu_z, sizeof(float)*GRID_SIZE, cudaMemcpyHostToDevice) );
src/cuda/src/gpu_astaroth.cu:        		printf("Inside substepGPU in gpu_astaroth &halo, &d_halo  %p %p pointing to  %p %p\n",&halo,&d_halo, halo, d_halo);
src/cuda/src/gpu_astaroth.cu:                	printf("Stop: Going to call copyouterhalostodevice inside gpu_astaroth.cu\n");
src/cuda/src/gpu_astaroth.cu:  cudaMemcpy(&lnrhoslice,d_lnrho+offset,mx*sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:  cudaMemcpy(&lnrhoslice,d_uu_x+offset,mx*sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&lnrhoslice,d_lnrho+offset,nx*sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu://cudaMemcpy(d_uu_y,&zero,mx*my*mz*sizeof(float),cudaMemcpyHostToDevice);
src/cuda/src/gpu_astaroth.cu://cudaMemcpy(d_uu_z,&zero,mx*my*mz*sizeof(float),cudaMemcpyHostToDevice);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val1, d_uu_x+offset, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val2, d_uu_x+offset+1, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val3, d_uu_x+offset+2, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val1, d_lnrho+offset, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val2, d_lnrho+offset+1, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val3, d_lnrho+offset+2, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:        checkErr(cudaMemcpyToSymbol(d_DT, &dt, sizeof(float)));
src/cuda/src/gpu_astaroth.cu:      //cudaMemcpy(&(lnrhoslice[kk]),d_lnrho+offset,sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:      cudaMemcpy(&(lnrhoslice[kk]),d_uu_x+offset,sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:        printf("Calling rungekutta2N_cuda\n");
src/cuda/src/gpu_astaroth.cu:	rungekutta2N_cuda(d_lnrho, d_uu_x, d_uu_y, d_uu_z, d_w_lnrho, d_w_uu_x, d_w_uu_y, d_w_uu_z,
src/cuda/src/gpu_astaroth.cu:      //cudaMemcpy(&(lnrhoslice[kk]),d_lnrho+offset,sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:      cudaMemcpy(&(lnrhoslice[kk]),d_uu_y+offset,sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val1, (void *) (d_uu_x+offset), sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val2, (void *) (d_uu_x+offset+1), sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val3, (void *) (d_uu_x+offset+2), sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val1, d_lnrho+offset, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val2, d_lnrho+offset+1, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:cudaMemcpy(&val3, d_lnrho+offset+2, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu://cudaMemcpy(d_uu_x,&zero,mx*my*mz*sizeof(float),cudaMemcpyHostToDevice);
src/cuda/src/gpu_astaroth.cu://cudaMemcpy(d_uu_y,&zero,mx*my*mz*sizeof(float),cudaMemcpyHostToDevice);
src/cuda/src/gpu_astaroth.cu://cudaMemcpy(d_uu_z,&zero,mx*my*mz*sizeof(float),cudaMemcpyHostToDevice);
src/cuda/src/gpu_astaroth.cu:/*cudaMemcpy(&lnrhoslice,d_lnrho+offset,mx*sizeof(float),cudaMemcpyDeviceToHost);
src/cuda/src/gpu_astaroth.cu:        if (ldiagnos) timeseries_diagnostics_cuda(it, dt, t);
src/cuda/src/gpu_astaroth.cu:                checkErr( cudaMemcpy(lnrho, d_lnrho, sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth.cu:                checkErr( cudaMemcpy(uu_x,  d_uu_x,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth.cu:                checkErr( cudaMemcpy(uu_y,  d_uu_y,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth.cu:                checkErr( cudaMemcpy(uu_z,  d_uu_z,  sizeof(float)*GRID_SIZE, cudaMemcpyDeviceToHost) );
src/cuda/src/gpu_astaroth.cu:extern "C" void finalizeGPU()
src/cuda/src/gpu_astaroth.cu:// Frees memory allocated on GPU.
src/cuda/src/gpu_astaroth.cu:        printf("stop1: inside finalizeGPU in gpy_astaroth.cu\n");
src/cuda/src/gpu_astaroth.cu:        //cudaEventDestroy( start );
src/cuda/src/gpu_astaroth.cu:        //cudaEventDestroy( stop );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_lnrho) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_uu_x) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_uu_y) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_uu_z) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_partial_result) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFree(d_halo) );
src/cuda/src/gpu_astaroth.cu:        /*checkErr( cudaFreeHost(slice_lnrho) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFreeHost(slice_uu) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFreeHost(slice_uu_x) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFreeHost(slice_uu_y) );
src/cuda/src/gpu_astaroth.cu:        checkErr( cudaFreeHost(slice_uu_z) );*/
src/cuda/src/gpu_astaroth.cu:        printf("stop2: inside finalizeGPU in gpy_astaroth.cu\n");
src/cuda/src/gpu_astaroth.cu:        cudaDeviceSynchronize();
src/cuda/src/gpu_astaroth.cu:        //checkErr(cudaDeviceReset());
src/cuda/src/gpu_astaroth.cu:        cudaDeviceReset();
src/cuda/src/gpu_astaroth.cu:        printf("GPU finalized %d", iproc);
src/cuda/src/collectiveops.cu://http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
src/cuda/src/collectiveops.cu:* 	-User is responsible for synchronizing the threads after calling if needed (cudaDeviceSynchronize())
src/cuda/src/collectiveops.cu:void max_vec_cuda(float* d_vec_max, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z)
src/cuda/src/collectiveops.cu:void min_vec_cuda(float* d_vec_min, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z)
src/cuda/src/collectiveops.cu:void vec_rms_cuda(float* d_vec_rms, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z, bool root=true)
src/cuda/src/collectiveops.cu:void max_scal_cuda(float* d_scal_max, float* d_partial_result, float* d_scal)
src/cuda/src/collectiveops.cu:void min_scal_cuda(float* d_scal_min, float* d_partial_result, float* d_scal)
src/cuda/src/collectiveops.cu:void scal_rms_cuda(float* d_scal_rms, float* d_partial_result, float* d_scal, bool sqr=true, bool root=true)
src/cuda/src/collectiveops.cu:void scal_exp_rms_cuda(float* d_scal_rms, float* d_partial_result, float* d_scal, bool sqr=true, bool root=true)
src/cuda/src/copyhalos.cuh:cudaError_t checkErr(cudaError_t result);
src/cuda/src/copyinternalhalostohost.cu:   Copying internal halos from GPU to host
src/cuda/src/copyinternalhalostohost.cu:	/*static cudaStream_t per_row_stream = NULL; 
src/cuda/src/copyinternalhalostohost.cu:		cudaStreamCreate(&per_row_stream);
src/cuda/src/copyinternalhalostohost.cu:	static cudaStream_t per_col_stream = NULL; 
src/cuda/src/copyinternalhalostohost.cu:		cudaStreamCreate(&per_col_stream);
src/cuda/src/copyinternalhalostohost.cu:	static cudaStream_t per_frtbk_stream = NULL; 
src/cuda/src/copyinternalhalostohost.cu:		cudaStreamCreate(&per_frtbk_stream);*/
src/cuda/src/copyinternalhalostohost.cu:	cudaDeviceSynchronize();
src/cuda/src/copyinternalhalostohost.cu:	cudaDeviceSynchronize();
src/cuda/src/copyinternalhalostohost.cu:	cudaDeviceSynchronize();
src/cuda/src/copyinternalhalostohost.cu:	//checkErr(cudaStreamDestroy(per_row_stream));
src/cuda/src/copyinternalhalostohost.cu:	//checkErr(cudaStreamDestroy(per_col_stream));
src/cuda/src/copyinternalhalostohost.cu:	//checkErr(cudaStreamDestroy(per_frtbk_stream));
src/cuda/src/collectiveops.cuh:void max_vec_cuda(float* d_vec_max, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z);
src/cuda/src/collectiveops.cuh:void min_vec_cuda(float* d_vec_min, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z);
src/cuda/src/collectiveops.cuh:void max_scal_cuda(float* d_scal_max, float* d_partial_result, float* d_scal);
src/cuda/src/collectiveops.cuh:void min_scal_cuda(float* d_scal_min, float* d_partial_result, float* d_scal);
src/cuda/src/collectiveops.cuh:void vec_rms_cuda(float* d_vec_rms, float* d_partial_result, float* d_vec_x, float* d_vec_y, float* d_vec_z, bool root=true);
src/cuda/src/collectiveops.cuh:void scal_rms_cuda(float* d_scal_rms, float* d_partial_result, float* d_scal, bool sqr=true, bool root=true);
src/cuda/src/collectiveops.cuh:void scal_exp_rms_cuda(float* d_scal_rms, float* d_partial_result, float* d_scal, bool sqr=true, bool root=true);
src/cuda/src/timestep.cu:float timestep_cuda(float* d_umax, float* d_partial_result, float* d_uu_x, float* d_uu_y, float* d_uu_z)
src/cuda/src/timestep.cu:        //MV: It is better to calculate dt within the CPU after we get umax from max_vec_cuda, 
src/cuda/src/timestep.cu:	max_vec_cuda(d_umax, d_partial_result, d_uu_x, d_uu_y, d_uu_z);
src/cuda/src/timestep.cu:	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:	cudaMemcpy(&umax, (float*)d_umax, sizeof(float), cudaMemcpyDeviceToHost); 
src/cuda/src/timestep.cu:	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:	max_scal_cuda(d_scaldiag, d_partial_result, d_src);
src/cuda/src/timestep.cu:       	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:       	cudaMemcpy(&maxscal, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:        min_scal_cuda(d_scaldiag, d_partial_result, d_src);
src/cuda/src/timestep.cu:        cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&minscal, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:        min_vec_cuda(d_scaldiag, d_partial_result, d_src_x, d_src_y, d_src_z);
src/cuda/src/timestep.cu:        cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&minvec, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:        max_vec_cuda(d_scaldiag, d_partial_result, d_src_x, d_src_y, d_src_z);
src/cuda/src/timestep.cu:        cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&maxvec, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:        vec_rms_cuda(d_scaldiag, d_partial_result, d_src_x, d_src_y, d_src_z);
src/cuda/src/timestep.cu:        cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&rmsvec, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:		scal_exp_rms_cuda(d_scaldiag, d_partial_result, d_src, true, false);
src/cuda/src/timestep.cu:		scal_rms_cuda(d_scaldiag, d_partial_result, d_src, true, false);
src/cuda/src/timestep.cu:	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&sumsquare, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:		scal_exp_rms_cuda(d_scaldiag, d_partial_result, d_src, false, false);
src/cuda/src/timestep.cu:		scal_rms_cuda(d_scaldiag, d_partial_result, d_src, false, false);
src/cuda/src/timestep.cu:	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:        cudaMemcpy(&sum, d_scaldiag, sizeof(float), cudaMemcpyDeviceToHost);
src/cuda/src/timestep.cu:	cudaDeviceSynchronize();
src/cuda/src/timestep.cu:void timeseries_diagnostics_cuda(int step, float dt, double t)
src/cuda/src/timestep.cu:	//Calculate and save all of the diagnostic variables calculated within the CUDA devices. 
src/entropy.f90:        if (ldt) then   !.not.lgpu) then
src/magnetic.f90:        if (.not.lgpu.and.ivid_poynting/=0) then
src/magnetic.f90:      if (.not.lgpu) then
src/magnetic.f90:      if (.not.lgpu) then
src/magnetic.f90:      if (.not.lgpu) then
src/magnetic.f90:      if (.not.lgpu) then
src/magnetic.f90:        if (.not.lgpu) then
src/magnetic.f90:        if (.not.lgpu) then
src/magnetic.f90:        if (.not.lgpu) then
src/magnetic.f90:        if (.not.lgpu) then
src/gpu.h:  public :: gpu_init, register_GPU, initialize_GPU, finalize_GPU, rhs_GPU, copy_farray_from_GPU, get_ptr_GPU
src/astaroth/gpu_astaroth.cc:/*                             gpu_astaroth.cc
src/astaroth/gpu_astaroth.cc:#define CUDA_ERRCHK(X)
src/astaroth/gpu_astaroth.cc://Do the 'isubstep'th integration step on all GPUs on the node and handle boundaries.
src/astaroth/gpu_astaroth.cc:extern "C" void substepGPU(int isubstep, bool full=false, bool early_finalize=false)
src/astaroth/gpu_astaroth.cc:         forcing_params.Update();  // calculate on CPU and load into GPU
src/astaroth/gpu_astaroth.cc:    //Integrate on the GPUs in this node
src/astaroth/gpu_astaroth.cc:      ERRCHK_CUDA_KERNEL_ALWAYS();
src/astaroth/gpu_astaroth.cc:extern "C" void registerGPU(AcReal* farray)
src/astaroth/gpu_astaroth.cc:extern "C" void initGPU()
src/astaroth/gpu_astaroth.cc:    // Initialize GPUs in the node
src/astaroth/gpu_astaroth.cc:extern "C" void initializeGPU(AcReal **farr_GPU_in, AcReal **farr_GPU_out)
src/astaroth/gpu_astaroth.cc:    //Setup configurations used for initializing and running the GPU code
src/astaroth/gpu_astaroth.cc:          *farr_GPU_in=p[0];
src/astaroth/gpu_astaroth.cc:          *farr_GPU_out=p[1];
src/astaroth/gpu_astaroth.cc:printf("From node layer: vbapointer= %p %p \n", *farr_GPU_in, *farr_GPU_out);
src/astaroth/gpu_astaroth.cc:          *farr_GPU_in=NULL;
src/astaroth/gpu_astaroth.cc:          *farr_GPU_out=NULL;
src/astaroth/gpu_astaroth.cc:extern "C" void finalizeGPU()
src/astaroth/gpu_astaroth.cc:    // Deallocate everything on the GPUs and reset
src/astaroth/Makefile.depend:gpu_astaroth.o: gpu_astaroth.cc $(AC_HEADERS) $(PCHEADER_DIR)/PC_moduleflags.h $(PCHEADER_DIR)/PC_module_parfuncs.h forcing.h loadStore.h $(CHEADERS)
src/astaroth/Makefile.depend:$(PCHEADER_DIR)/timeseries_$(PCHEADER_DIR).o: $(PCHEADER_DIR)/timeseries_$(PCHEADER_DIR).cc $(PCHEADER_DIR)/grid.h $(PCHEADER_DIR)/qualify.h utils/utils.h gpu/cuda/cuda_generic.cuh ../cparam_c.h ../cdata_c.h ../$(PCHEADER_DIR)_c.h $(PCHEADER_DIR)/PC_module_diagfuncs.h $(PCHEADER_DIR)/PC_modulediags_init.h $(PCHEADER_DIR)/PC_modulediags.h
src/astaroth/Makefile.depend:gpu/cuda/generic/collectiveops_cuda_generic.o: gpu/cuda/generic/collectiveops_cuda_generic.cu gpu/cuda/generic/collectiveops_cuda_generic.cuh gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh utils/utils.h $(PCHEADER_DIR)/errorhandler.h $(PCHEADER_DIR)/config.h
src/astaroth/Makefile.depend:gpu/cuda/core/copyHalosConcur.o: gpu/cuda/core/copyHalosConcur.cu gpu/cuda/core/copyHalosConcur.cuh gpu/cuda/cuda_generic.cuh $(PCHEADER_DIR)/PC_moduleflags.h $(PCHEADER_DIR)/PC_modulepardecs.h $(PCHEADER_DIR)/datatypes.h $(PCHEADER_DIR)/errorhandler.h gpu/cuda/core/dconsts_core.cuh ../cdata_c.h
src/astaroth/Makefile.depend:gpu/cuda/core/cuda_core.o: gpu/cuda/core/cuda_core.cu gpu/cuda/core/cuda_core.cuh gpu/cuda/cuda_generic.cuh $(PCHEADER_DIR)/datatypes.h $(PCHEADER_DIR)/errorhandler.h gpu/cuda/core/dconsts_core.cuh gpu/cuda/core/errorhandler_cuda.cuh gpu/cuda/core/copyHalosConcur.cuh $(PCHEADER_DIR)/PC_moduleflags.h $(PCHEADER_DIR)/PC_modulepardecs.h $(PCHEADER_DIR)/config.h $(PCHEADER_DIR)/grid.h $(PCHEADER_DIR)/slice.h $(PCHEADER_DIR)/forcing.h $(PCHEADER_DIR)/PC_moduleflags.h $(PCHEADER_DIR)/PC_modulepars.h $(PCHEADER_DIR)/PC_module_parfuncs.h
src/astaroth/Makefile:# Settings for taito-gpu
src/astaroth/Makefile:#  1) cuda/9.0   3) openmpi/2.0.1_ic16.0            5) hdf5/1.8.16_openmpi_2.0.1_ic16.0 7) gcc/5.3.0
src/astaroth/Makefile:#  2) intel/2016 4) fftw/2.1.5_openmpi_2.0.1_ic16.0 6) cuda/9.1
src/astaroth/Makefile:SOURCES = gpu_astaroth.cc
src/astaroth/Makefile:#GCC with CUDA
src/astaroth/Makefile:#export CUDA_NVCC_FLAGS='-shared -Xcompiler -fPIC'
src/astaroth/Makefile:#	cmake -DMULTIGPU_ENABLED=OFF -DDSL_MODULE_DIR=../acc-runtime/$(DSL_MODULE_DIR) -DUSE_OMP=ON -DMPI_ENABLED=OFF -DBUILD_SHARED_LIBS=ON -DSINGLEPASS_INTEGRATION=OFF -DUSE_CUDA_AWARE_MPI=OFF -DUSE_EXTERNAL_DECOMP=ON -DLINEAR_PROC_MAPPING=ON -DBUILD_SAMPLES=OFF -DBUILD_STANDALONE=OFF ..
src/astaroth/Makefile:	cmake -DMULTIGPU_ENABLED=OFF -DDSL_MODULE_DIR=../acc-runtime/$(DSL_MODULE_DIR) -DUSE_OMP=ON -DMPI_ENABLED=OFF -DBUILD_SHARED=ON -DSINGLEPASS_INTEGRATION=$(SINGLEPASS_INTEGRATION) -DUSE_CUDA_AWARE_MPI=OFF\
src/astaroth/gpu_astaroth_cpubcs.cc:/*                             gpu_astaroth.cc
src/astaroth/gpu_astaroth_cpubcs.cc:#define CUDA_ERRCHK(X)
src/astaroth/gpu_astaroth_cpubcs.cc://Do the 'isubstep'th integration step on all GPUs on the node and handle boundaries.
src/astaroth/gpu_astaroth_cpubcs.cc:extern "C" void substepGPU(int isubstep, bool full=false, bool early_finalize=false)
src/astaroth/gpu_astaroth_cpubcs.cc:         forcing_params.Update();  // calculate on CPU and load into GPU
src/astaroth/gpu_astaroth_cpubcs.cc:    //Integrate on the GPUs in this node
src/astaroth/gpu_astaroth_cpubcs.cc:      ERRCHK_CUDA_KERNEL_ALWAYS();
src/astaroth/gpu_astaroth_cpubcs.cc:extern "C" void registerGPU(AcReal* farray)
src/astaroth/gpu_astaroth_cpubcs.cc:extern "C" void initGPU()
src/astaroth/gpu_astaroth_cpubcs.cc:    // Initialize GPUs in the node
src/astaroth/gpu_astaroth_cpubcs.cc:extern "C" void initializeGPU(AcReal **farr_GPU_in, AcReal **farr_GPU_out)
src/astaroth/gpu_astaroth_cpubcs.cc:    //Setup configurations used for initializing and running the GPU code
src/astaroth/gpu_astaroth_cpubcs.cc:          *farr_GPU_in=p[0];
src/astaroth/gpu_astaroth_cpubcs.cc:          *farr_GPU_out=p[1];
src/astaroth/gpu_astaroth_cpubcs.cc:printf("From node layer: vbapointer= %p %p \n", *farr_GPU_in, *farr_GPU_out);
src/astaroth/gpu_astaroth_cpubcs.cc:          *farr_GPU_in=NULL;
src/astaroth/gpu_astaroth_cpubcs.cc:          *farr_GPU_out=NULL;
src/astaroth/gpu_astaroth_cpubcs.cc:extern "C" void finalizeGPU()
src/astaroth/gpu_astaroth_cpubcs.cc:    // Deallocate everything on the GPUs and reset
src/astaroth/loadStore.cc:        //!!!cudaHostRegister(mesh, size, cudaHostRegisterDefault);    // time-critical!
src/astaroth/loadStore.cc:        //!!!cudaHostRegister(mesh, size, cudaHostRegisterDefault);    // time-critical!
src/astaroth/README_install.md:  - git clone -b gputestv6 --recurse-submodules https://<username>@github.com/pencil-code/pencil-code.git
src/astaroth/README_install.md:  - git clone -b gputestv6 --recurse-submodules https://<username>@pencil-code.org/git/ pencil-code
src/astaroth/README_install.md:  git checkout gputestv6

```
