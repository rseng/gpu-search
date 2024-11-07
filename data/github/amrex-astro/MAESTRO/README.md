# https://github.com/AMReX-Astro/MAESTRO

```console
Docs/SDC/flowchart_SDC.eps:				/currentcolorspace exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/sep_colorspace_dict null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:		/AGMCORE_gput{
Docs/SDC/flowchart_SDC.eps:			dup/AGMCORE_currentoverprint exch AGMCORE_gput AGMCORE_&setoverprint
Docs/SDC/flowchart_SDC.eps:	/currentcmykcolor[0 0 0 0]AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/currentstrokeadjust false AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/currentcolorspace[/DeviceGray]AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/sep_tint 0 AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/devicen_tints[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/sep_colorspace_dict null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/devicen_colorspace_dict null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/indexed_colorspace_dict null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/currentcolor_intent()AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/customcolor_tint 1 AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/absolute_colorimetric_crd null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/relative_colorimetric_crd null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/saturation_crd null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/perceptual_crd null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:		4 copy AGMCORE_cmykbuf astore/currentcmykcolor exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:									/sep_colorspace_dict currentdict AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:			/currentstrokeadjust exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:			/currentcolorspace exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:		/currentcolorspace[/DeviceCMYK]/AGMCORE_gput cvx
Docs/SDC/flowchart_SDC.eps:	/sep_colorspace_dict null AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:		dup/sep_tint exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	dup/sep_colorspace_dict exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	dup/devicen_colorspace_dict exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	dup/indexed_colorspace_dict exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:					/absolute_colorimetric_crd exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:						/relative_colorimetric_crd exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:							/saturation_crd exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:								/perceptual_crd exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:	/customcolor_tint 1 AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:		dup/customcolor_tint exch AGMCORE_gput
Docs/SDC/flowchart_SDC.eps:			not{/sep_tint 1.0 AGMCORE_gput}if
Docs/managing_jobs/managingjobs.tex:\subsection{Profiling and Debugging on GPUs}
Docs/managing_jobs/managingjobs.tex:To get an idea of how code performs on Titan's GPUs, there are a few tools
Docs/managing_jobs/managingjobs.tex:codes that run on several nodes, not specifically for analyzing GPU usage.
Docs/managing_jobs/managingjobs.tex:Still, they do support some GPU analysis.  In the next section, we'll discuss
Docs/managing_jobs/managingjobs.tex:NVIDA's tools specifically for analyzing GPU usage.
Docs/managing_jobs/managingjobs.tex:$ module load cudatoolkit
Docs/managing_jobs/managingjobs.tex:support for analyzing OpenACC code.  The next step is to compile.  You
Docs/managing_jobs/managingjobs.tex:$ scorep --cuda --openacc -v ftn gpuprogram.f90
Docs/managing_jobs/managingjobs.tex:export SCOREP_CUDA_ENABLE=yes,kernel_counter,flushatexit
Docs/managing_jobs/managingjobs.tex:export SCOREP_CUDA_BUFFER=200M
Docs/managing_jobs/managingjobs.tex:export SCOREP_OPENACC_ENABLE=yes
Docs/managing_jobs/managingjobs.tex:NVIDIA provides tools for specifically analyzing how your code utilizes their
Docs/managing_jobs/managingjobs.tex:GPUs.  {\tt Score-P} is a fully-featured profiler with some CUDA and OpenACC
Docs/managing_jobs/managingjobs.tex:support.  It can be useful for providing context for GPU execution and it allows
Docs/managing_jobs/managingjobs.tex:you to, for example, see line numbers for OpenACC directives that are executed.
Docs/managing_jobs/managingjobs.tex:{\tt nvprof} will only analyze GPU execution, but in exchange you get much more
Docs/managing_jobs/managingjobs.tex:detail than is available with {\tt Score-P}.  {\tt nvvp} is NVIDIA's visual
Docs/managing_jobs/managingjobs.tex:is the guided analysis it will perform, which analyzes your code's GPU
Docs/managing_jobs/managingjobs.tex:provided when you load the {\tt cudatoolkit} module.
Docs/managing_jobs/managingjobs.tex:$ aprun -b nvprof --profile-child-processes ./gpuprogram.exe arg1 arg2... 
Docs/managing_jobs/managingjobs.tex:  ./gpuprogram.exe arg1 arg2... 
Docs/managing_jobs/managingjobs.tex:  -o nvprof.metrics.out%p ./gpuprogram.exe arg1 arg2... 
Docs/managing_jobs/managingjobs.tex:performance is a spreadsheet developed by NVIDIA to calculate occupancy.  Every
Docs/managing_jobs/managingjobs.tex:installation of the CUDA Toolkit should have this occupancy calculator in a
Docs/managing_jobs/managingjobs.tex:\url{http://developer.download.nvidia.com/compute/cuda/CUDA_Occupancy_calculator.xls}.
Docs/managing_jobs/managingjobs.tex:bit of interesting insight into optimizing a GPU code.  More on occupancy can be
Docs/managing_jobs/managingjobs.tex:\url{http://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy}.
Docs/managing_jobs/managingjobs.tex:and an NVIDIA GK110 ``Kepler'' GPU accelerator (Tesla K20X).  The
Docs/managing_jobs/managingjobs.tex:GPU is configured with 14 streaming multiprocessor units (SMXs), each
Docs/managing_jobs/managingjobs.tex:of which has 192 single-precision or 64 double-precision CUDA cores.  Thus
Docs/managing_jobs/managingjobs.tex:there are a total of 2688 SP CUDA cores or 896 DP CUDA cores.
Docs/unit_tests/test_project_3d.eps:(u"Z9%,ICoL$bY2PY_ui2GPug!$%8)kSg0l)\Q+-b;p'ZjNO,XZA&n4A618BI0`\h_u4]1)L]Ki+/R
Docs/unit_tests/test_project_3d.eps:K@Z*qHpBnK9<+4F9'!T[hua'=dee%['roCmV7nC7;QU?q5Y.FFPaDBY+QU:Mm<uP^pHJ7fPf-r0b%6
Docs/unit_tests/test_project_3d.eps:[`gJen.'XQ7?h(M[GPu:g7e(&:lYL&5k??l\3!T]>p7h>\G!c71mh52fVi8u@;b',sK[Y"Mn4/oh%i
Docs/unit_tests/test_project_3d.eps:<$r+Zrq;Wj@fb78)9[b5KAc`JgpUNYU6R(k9#c76+n\"^5[Ng4fi1klldB.KJY@88u?*dG4J+C++1)
Docs/unit_tests/test_project_3d.eps:VfNC/>rocm2tVl-q@3=`FEI3!a3E<=jY52nJ_.U=`8\HuI%5#\]/&(^qS)/\%B(\=-.VBo/QD=Q;gU
Docs/unit_tests/test_project_3d.eps:-t"Kf]shVaU$,83l^<!2G<WbS5s<'o\+r<)3b1,t^.77egpuUEACo"$choTVqhk8/7B$9B-s"J>W9>
Docs/unit_tests/test_project_3d.eps:L1oC>BjF?dr##_N%93GpuQAerTu2jA4rCdCCf$H#h*ST$b[Ho]7.I`M%O[:3B0E28.[f7U[k(B\_Dn
Docs/unit_tests/test_project_3d.eps:V#QDnHQ]GPu=FYWu6q;Gmu3.&e=5KPsATtHd."22hjM;a(/b+,T-I']Ld\')/,146p^6+NaQbG1j\U
Docs/unit_tests/test_project_3d.eps:04sE`lJG!=oNQ<+5_C/M?7lSGc-"@(Gpu+.VjODJ:gW,#O+N[NeO$65Wi20j=QEHH2LXFXr8W8Uj^a
Docs/unit_tests/test_project_3d.eps:X!FV8?1ZX3p;g@^h9M]0MOhM6K#i`$l3pZ6,/.24?P<63o3kFiBg3Y(3P_ciSh:NWA:23&(715gpuu
Docs/unit_tests/test_project_3d.eps:b%Vi!d!!8drWN#*kgM2A'WjO1T1MLU!P)jA2-TpNcclhU%>-+8#?C;sZJO0nnqE<r"=AC%#DK0"1KJ
Util/VBDF/bdf.f90:    !but for GPU dev I'm trying to simplify.
Util/VBDF/bdf.f90:       !TODO: Debug I/O not cool on GPUs. If we want to keep it, need to rewrite
Util/VBDF/bdf.f90:       !TODO: cycle statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/bdf.f90:          !TODO: exit statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/bdf.f90:    !TODO: GPUs don't like print statements.  Either delete this or work up alternative implementations
Util/VBDF/bdf.f90:                !   allowed on GPUs
Util/VBDF/bdf.f90:          !   on the GPUs.  Assignments of form array = array + expr often
Util/VBDF/bdf.f90:          !   for GPUs.
Util/VBDF/bdf.f90:  !on the GPU.  Gotta do it the less pretty way.
Util/VBDF/bdf.f90:  ! A local, GPU-compiled version of intrinsic eoshift 
Util/VBDF/bdf.f90:  ! NOTE: Array-valued functions are NOT allowed on the GPU (in PGI at least), had to rewrite this
Util/VBDF/bdf.f90:  ! A local, GPU-compiled version of intrinsic minloc
Util/VBDF/bdf.f90:  ! TODO: Check if this is implemented on GPU, if so delete all this
Util/VBDF/bdf.f90:        !using optimized.  Can't use Fortran intrinsic matmul() on GPU
Util/VBDF/oac.bdf.f90:    !but for GPU dev I'm trying to simplify.
Util/VBDF/oac.bdf.f90:       !TODO: Debug I/O not cool on GPUs. If we want to keep it, need to rewrite
Util/VBDF/oac.bdf.f90:       !TODO: cycle statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/oac.bdf.f90:       !TODO: exit statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/oac.bdf.f90:    !TODO: GPUs don't like print statements.  Either delete this or work up alternative implementations
Util/VBDF/oac.bdf.f90:                !   allowed on GPUs
Util/VBDF/oac.bdf.f90:          !   on the GPUs.  Assignments of form array = array + expr often
Util/VBDF/oac.bdf.f90:          !   for GPUs.
Util/VBDF/oac.bdf.f90:  ! A local, GPU-compiled version of intrinsic eoshift 
Util/VBDF/oac.bdf.f90:  ! NOTE: Array-valued functions are NOT allowed on the GPU (in PGI at least), had to rewrite this
Util/VBDF/oac.bdf.f90:  ! A local, GPU-compiled version of intrinsic minloc
Util/VBDF/oac.bdf.f90:  ! TODO: Check if this is implemented on GPU, if so delete all this
Util/VBDF/dev/t1_oac_simp/bdf.f90:    !but for GPU dev I'm trying to simplify.
Util/VBDF/dev/t1_oac_simp/bdf.f90:       !TODO: Debug I/O not cool on GPUs. If we want to keep it, need to rewrite
Util/VBDF/dev/t1_oac_simp/bdf.f90:       !TODO: cycle statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/dev/t1_oac_simp/bdf.f90:       !TODO: exit statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/dev/t1_oac_simp/bdf.f90:    !TODO: GPUs don't like print statements.  Either delete this or work up alternative implementations
Util/VBDF/dev/t1_oac_simp/bdf.f90:                !   allowed on GPUs
Util/VBDF/dev/t1_oac_simp/bdf.f90:          !   on the GPUs.  Assignments of form array = array + expr often
Util/VBDF/dev/t1_oac_simp/bdf.f90:          !   for GPUs.
Util/VBDF/dev/t1_oac_simp/bdf.f90:  ! A local, GPU-compiled version of intrinsic eoshift 
Util/VBDF/dev/t1_oac_simp/bdf.f90:  ! NOTE: Array-valued functions are NOT allowed on the GPU (in PGI at least), had to rewrite this
Util/VBDF/dev/t1_oac_simp/bdf.f90:  ! A local, GPU-compiled version of intrinsic minloc
Util/VBDF/dev/t1_oac_simp/bdf.f90:  ! TODO: Check if this is implemented on GPU, if so delete all this
Util/VBDF/dev/t1_oac_simp/Makefile:    FFLAGS  = -module build -Ibuild -acc -Minfo=acc -Mcuda=cuda7.0 -ta=nvidia:maxwell
Util/VBDF/dev/t1_oac_simp/Makefile:    #You can use this set of flags to play with PGI's beta implementation of managed memory on NVIDIA cards	
Util/VBDF/dev/t1_oac_simp/Makefile:	 #FFLAGS  = -module build -Ibuild -acc -Minfo=acc -Mcuda=cuda7.0 -ta=nvidia:maxwell,managed
Util/VBDF/dev/t1_oac_simp/Makefile:  #   cudatoolkit
Util/VBDF/dev/t1_oac_simp/Makefile:    FFLAGS  = -Ibuild -Jbuild -h msgs -h acc -lcudart
Util/VBDF/dev/t1_oac_simp/t1.f90:! This version has been adapted for developing and testing OpenACC acceleration
Util/VBDF/dev/t1_oac_simp/t1.f90:   !Have the GPU loop over state data, with the intention of having each
Util/VBDF/dev/t1_oac_simp/t1.f90:   !CUDA core execute the acc seq routine bdf_advance on a cell of hydro data
Util/VBDF/dev/t1_oac_simp/deepcopy_demo.f90:!$ pgf95 -acc -Minfo=acc -Mcuda=cuda7.0 -ta=nvidia:maxwell deepcopy_demo.f90
Util/VBDF/dev/t1_oac_simp/README:bdf*, came with t1.  This version uses vbdf** with OpenACC to target
Util/VBDF/dev/t1_oac_simp/README:accelerating co-processors like GPUs.  It runs the same t1 problem on all CUDA
Util/VBDF/dev/t1_oac_simp/deadbeef.F:!     You can made CUDA Fortran programs like this with 
Util/VBDF/dev/t1_oac_simp/deadbeef.F:!     $ pgf90 -Mcuda deadbeef.F
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      Use CudaFor
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      iStat = cudaMemGetInfo(FreeMem, TotGlbMem)
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      iStat = cudaMemSet(A,iSet,N)
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      If(IsKrnOK(' Problem with cudaMemSet')) then
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      Use CUDAfor
Util/VBDF/dev/t1_oac_simp/deadbeef.F:C     Checks for (A)Syncronous errors from CUDA kernel launches
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      iErrS = cudaGetLastError()
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      iErrA = cudaDeviceSynchronize()
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      If(iErrS.ne.cudaSuccess)  then
Util/VBDF/dev/t1_oac_simp/deadbeef.F:     $            cudaGetErrorString(iErrS)
Util/VBDF/dev/t1_oac_simp/deadbeef.F:      If(iErrA.ne.cudaSuccess)  then
Util/VBDF/dev/t1_oac_simp/deadbeef.F:     $            cudaGetErrorString(cudaGetLastError())
Util/VBDF/dev/oac_linalg/tdaxpy.sh:pgf90 -o tdaxpy_gpu \
Util/VBDF/dev/oac_linalg/tdaxpy.sh:  -fast -acc -ta=tesla -Mcuda=cuda7.0 -Minfo=accel \
Util/VBDF/dev/omp_vode/test_react/react_state.f90:  !burner/integrator.  Useful for burners that target GPUs.
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:! My understanding of GPU/OpenACC constructs (subject to ignorance): 
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:!   vector --> vector of threads, typically a CUDA block.  A block of threads
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:!   executes in lock-step on data.  gang --> CUDA grid.  Grid is a group of blocks,
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:! VBDF-style vectorized RHS with OpenACC acceleration
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:  ! These are only needed to placate the restrictions of OpenACC
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:  !OpenACC guidelines: 
Util/VBDF/dev/omp_vode/ignition_simple_bdf/f_rhs.f90:  !  OpenACC/Cray's automatic scoping often leads
Util/VBDF/dev/vectorized/test_react/react_state.f90:  !burner/integrator.  Useful for burners that target GPUs.
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:             !TODO: Compile on GPU?
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:          !TODO: GPU?
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:             !TODO: Compile on GPU?
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:       !TODO: Compile on GPU
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:       !TODO: GPU
Util/VBDF/dev/vectorized/test_react/bdf_semivec.f90:          !TODO: Compile on GPU
Util/VBDF/dev/vectorized/ignition_simple_bdf/f_rhs.f90:! My understanding of GPU/OpenACC constructs (subject to ignorance): 
Util/VBDF/dev/vectorized/ignition_simple_bdf/f_rhs.f90:!   vector --> vector of threads, typically a CUDA block.  A block of threads
Util/VBDF/dev/vectorized/ignition_simple_bdf/f_rhs.f90:!   executes in lock-step on data.  gang --> CUDA grid.  Grid is a group of blocks,
Util/VBDF/dev/vectorized/ignition_simple_bdf/f_rhs.f90:    ! TODO: OpenACC-ify this routine
Util/VBDF/dev/vectorized/README:Maestro-compatible and with some further vectorization, but no OpenACC
Util/VBDF/dev/t1_oac/bdf.f90:  ! A local, GPU-compiled version of intrinsic eoshift 
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:Some notes on using OPENACC.
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:OLCF has a nice GPU guide:                https://www.olcf.ornl.gov/support/system-user-guides/accelerated-computing-guide/
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:And there are some Blue Waters resources: https://bluewaters.ncsa.illinois.edu/openacc
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:The BoxLib/Maestro build system already knows about OpenACC, so you just
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:craype-accel-nvidia35
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:cudatoolkit
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:(e.g. '$ module load craype-accel-nvidia35)
Util/VBDF/dev/t1_oac/OPENACC.bw.txt:In your batch script, be sure you request xk nodes (the ones with GPUs), similar to:
Util/VBDF/dev/omp_vbdf/test_react/react_state.f90:  !burner/integrator.  Useful for burners that target GPUs.
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/temp.burner.f90:      ! this, so this is just a temporary hack for the purposes of rapid GPU development.
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/temp.burner.f90:         !We need to build, allocate bdf_ts objects before the OpenACC region,
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/temp.burner.f90:         !because you cannot allocate within OpenACC regions.
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/temp.burner.f90:      !TODO: Put first OpenACC parallel here with a loop over vector inputs
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/burner.f90:      ! this, so this is just a temporary hack for the purposes of rapid GPU development.
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/burner.f90:         !We need to build, allocate bdf_ts objects before the OpenACC region,
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/burner.f90:         !because you cannot allocate within OpenACC regions.
Util/VBDF/dev/omp_vbdf/ignition_simple_bdf/burner.f90:      !TODO: Put first OpenACC parallel here with a loop over vector inputs
Util/VBDF/dev/oac_vbdf/test_react/react_state.f90:  !burner/integrator.  Useful for burners that target GPUs.
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:    !but for GPU dev I'm trying to simplify.
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:       !TODO: Debug I/O not cool on GPUs. If we want to keep it, need to rewrite
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:       !TODO: cycle statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:       !TODO: exit statements may lead to bad use of coalesced memory in OpenACC (or busy waiting),
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:    !TODO: GPUs don't like print statements.  Either delete this or work up alternative implementations
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:                !   allowed on GPUs
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:          !   on the GPUs.  Assignments of form array = array + expr often
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:          !   for GPUs.
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:  ! A local, GPU-compiled version of intrinsic eoshift 
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:  ! NOTE: Array-valued functions are NOT allowed on the GPU (in PGI at least), had to rewrite this
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:  ! A local, GPU-compiled version of intrinsic minloc
Util/VBDF/dev/oac_vbdf/test_react/bdf.f90:  ! TODO: Check if this is implemented on GPU, if so delete all this
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:      ! this, so this is just a temporary hack for the purposes of rapid GPU development.
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:      !      and then updating serves to overwrite the device (GPU) pointer with
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:         !We need to build and allocate bdf_ts objects before the OpenACC region,
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:         !because you cannot allocate within OpenACC regions.
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:         !address with that of the host, according to PGI/NVIDIA consults.
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:            !because such operations often cause errors on GPU.
Util/VBDF/dev/oac_vbdf/ignition_simple_bdf/burner.f90:         print *, 'sum(ierr) for all GPU threads: ', ierr_tot
Util/VBDF/dev/oac_vbdf/README:This is a GPU- (OpenACC-)accelerated version of the test_react Maestro unit test using
Util/VBDF/mae.bdf.f90:             !TODO: Compile on GPU?
Util/VBDF/mae.bdf.f90:          !TODO: GPU?
Util/VBDF/mae.bdf.f90:                   !TODO: Compile on GPU?
Util/VBDF/mae.bdf.f90:       !TODO: Compile on GPU
Util/VBDF/mae.bdf.f90:       !TODO: GPU
Util/VBDF/mae.bdf.f90:          !TODO: Compile on GPU
Util/VBDF/README:making it ideal for targeting GPUs.
Util/VBDF/README:oac.bdf.f90 : A version with OpenACC directives, enabling the integrator to be
Util/VBDF/README:              run on co-processors like GPUs. Developed by Adam Jacobs.
Util/postprocessing/urca-tools/GNUmakefile:# we are not using the CUDA stuff
Util/job_scripts/bw/bw.run:# for GPU).  Note that each (CPU) node has 32 integer cores, every two of which share a
Util/job_scripts/bw/bw.bash.run:# for GPU).  Note that each (CPU/xe) node has 32 integer cores, every two of which share a
Util/job_scripts/bw/bw.bash.run:# single floating point core (GPU/xk nodes have 16 integer cores).  By specifying ppn=32, we 
Util/job_scripts/bw/bw.bash.run:#   1 NVIDIA GK110 "Kepler" K20X accelerator
Util/job_scripts/bw/bw.bash.run:#     Each SMX has 192 single-precision CUDA cores, or 64 double-precision CUDA cores
GMaestro.mak:# we are not using the CUDA stuff
GMaestro.mak:# If we are using OpenACC, add the corresponding preprocessor macro.
Microphysics/networks/burn_type.F90:  ! Implement a manual copy routine since CUDA Fortran doesn't
Microphysics/networks/ignition_simple_bdf/TODO:+ Add tests similar to those in ignition_simple for this OpenACC/bdf version.
Microphysics/networks/ignition_simple_bdf/README:modified by Adam Jacobs to work with Maestro and on GPUs via OpenACC.
Microphysics/networks/ignition_simple_bdf/burner.f90:      ! this, so this is just a temporary hack for the purposes of rapid GPU development.
Microphysics/networks/ignition_simple_bdf/burner.f90:         !because such operations often cause errors on GPU.
Microphysics/networks/ignition_simple_bdf/burner.f90:      !   print *, 'sum(ierr) for all GPU threads: ', ierr_tot
Source/probin.template:    ! Update the value of the parameters on the GPU.

```
