# https://github.com/lwang-astro/PeTar

```console
sample/star_cluster_bse_galpy.sh:# The initial condition for NBODY6++GPU is created with -C 5, this is used to generated initial condtion for PeTar
sample/star_cluster_bse.sh:# The initial condition for NBODY6++GPU is created with -C 5, this is used to generated initial condtion for PeTar
sample/star_cluster.sh:# The initial condition for NBODY6++GPU is created with -C 5, this is used to generated initial condtion for PeTar
configure.ac:		       CUDAFLAGS=$CUDAFLAGS" -std=c++"$with_cpp_std],
configure.ac:# cuda
configure.ac:AC_ARG_ENABLE([cuda],
configure.ac:              [AS_HELP_STRING([--enable-cuda],
configure.ac:                              [enable CUDA (GPU) acceleration support for long-distant tree force])],
configure.ac:              [use_cuda=yes],
configure.ac:              [use_cuda=no])
configure.ac:AC_ARG_WITH([cuda-prefix], 
configure.ac:            [AS_HELP_STRING([--with-cuda-prefix],
configure.ac:	                    [Prefix of your CUDA installation])],
configure.ac:	    [cuda_prefix="/usr/local/cuda"])
configure.ac:#AC_ARG_WITH([cuda-sdk-prefix], 
configure.ac:#            [AS_HELP_STRING([--with-cuda-sdk-prefix],
configure.ac:# 	                    [Prefix of your CUDA samples (SDK) installation])],
configure.ac:#            [cuda_sdk_prefix=$withval],
configure.ac:# 	    [cuda_sdk_prefix="/usr/local/cuda/samples"])
configure.ac:AS_IF([test "x$use_cuda" != xno],
configure.ac:       use_cuda=yes
configure.ac:             [AC_PATH_PROG([NVCC], [nvcc], [none], [$cuda_prefix/bin])
configure.ac:                    [AC_MSG_FAILURE([can't find CUDA compiler nvcc, please check whether nvcc is in environment PATH or use --with-cuda-prefix to provide the PATH of CUDA installation])])],
configure.ac:              cuda_prefix=`which nvcc|sed 's:/bin/nvcc::'`])
configure.ac:       PROG_NAME=$PROG_NAME".gpu"
configure.ac:       AC_MSG_CHECKING([CUDA version])
configure.ac:       CUDA_VERSION=`$NVCC --version|awk -F ',' 'END{print $(NF-1)}'|awk '{print $2}'`
configure.ac:       AC_MSG_RESULT([$CUDA_VERSION])
configure.ac:#       AC_CHECK_FILE(["$cuda_prefix/samples/common/inc/helper_cuda.h"],
configure.ac:#                     [CUDAFLAGS=$CUDAFLAGS" -I $cuda_prefix/samples/common/inc"],
configure.ac:#                     [AC_CHECK_FILE(["$cuda_sdk_prefix/common/inc/helper_cuda.h"],
configure.ac:#                                    [CUDAFLAGS=$CUDAFLAGS" -I $cuda_sdk_prefix/common/inc"],
configure.ac:#                                    [AC_CHECK_FILE(["$cuda_sdk_prefix/Common/helper_cuda.h"],
configure.ac:#                                                   [CUDAFLAGS=$CUDAFLAGS" -I $cuda_sdk_prefix/Common"],
configure.ac:#                                                   [ AC_MSG_FAILURE([can't find CUDA sample inc file helper_cuda.h, please provide correct cuda SDK prefix by using --with-cuda-sdk-prefix])])])])
configure.ac:       AC_CHECK_LIB([cudart],
configure.ac:                    [CUDALIBS=' -lcudart'],
configure.ac:                    [AC_CHECK_FILE(["$cuda_prefix/lib64/libcudart.so"],
configure.ac:                                   [CUDALIBS=" -L$cuda_prefix/lib64 -lcudart"],
configure.ac:                                   [AC_MSG_FAILURE([can't find CUDA library -lcudart, please provide correct cuda PREFIX by using --with-cuda-prefix])])])
configure.ac:		    [CUDALIBS=$CUDALIBS' -lgomp'],
configure.ac:		    [AC_CHECK_FILE(["$cuda_prefix/lib64/libgomp.so"],
configure.ac:                                   [CUDALIBS=$CUDALIBS" -lgomp"],
configure.ac:                                   [AC_MSG_FAILURE([can't find CUDA library -lgomp, please provide correct cuda PREFIX by using --with-cuda-prefix])])])],
configure.ac:       [use_cuda=no])
configure.ac:AC_SUBST([use_cuda])
configure.ac:AC_SUBST([CUDAFLAGS])
configure.ac:AC_SUBST([CUDALIBS])
configure.ac:AC_MSG_NOTICE([     Using GPU:         $use_cuda])
configure.ac:AS_IF([test "x$use_cuda" != xno],
configure.ac:      [AC_MSG_NOTICE([     CUDA compiler:     $NVCC])
configure.ac:       AC_MSG_NOTICE([     CUDA version:      $CUDA_VERSION])])
codemeta.json:    "description": "The N-body code PETAR (ParticlE Tree & particle-particle & Algorithmic Regularization) combines the methods of Barnes-Hut tree, Hermite integrator and slow-down algorithmic regularization (SDAR). It accurately handles an arbitrary fraction of multiple systems (<i>e.g.</i> binaries, triples) while keeping a high performance by using the hybrid parallelization methods with MPI, OpenMP, SIMD instructions and GPU. PETAR has very good agreement with NBODY6++GPU results on the long-term evolution of the global structure, binary orbits and escapers and is significantly faster when used on a highly configured GPU desktop computer. PETAR scales well when the number of cores increase on the Cray XC50 supercomputer, allowing a solution to the ten million-body problem which covers the region of ultra compact dwarfs and nuclear star clusters.",
doc/html/artificial__particles_8hpp__dep__incl.map:<area shape="rect" id="node14" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="170,379,322,405"/>
doc/html/force__gpu__cuda_8hpp.js:var force__gpu__cuda_8hpp =
doc/html/force__gpu__cuda_8hpp.js:    [ "CalcForceWithLinearCutoffCUDA", "structCalcForceWithLinearCutoffCUDA.html", "structCalcForceWithLinearCutoffCUDA" ],
doc/html/force__gpu__cuda_8hpp.js:    [ "SPJSoft", "force__gpu__cuda_8hpp.html#aaacf9eee49e4260d214d9be1f494a194", null ],
doc/html/force__gpu__cuda_8hpp.js:    [ "RetrieveForceCUDA", "force__gpu__cuda_8hpp.html#aedbe85337286f7c675e64a36cb78d099", null ]
doc/html/simd__test_8cxx_a3c04138a5bfe5d72780bb7e82a18e627_cgraph.map:<area shape="rect" id="node6" href="$force__gpu__cuda_8hpp.html#aedbe85337286f7c675e64a36cb78d099" title=" " alt="" coords="138,224,289,251"/>
doc/html/inherit_graph_15.map:<area shape="rect" id="node1" href="$structcudaPointer.html" title=" " alt="" coords="5,5,143,32"/>
doc/html/files_dup.js:    [ "cuda_pointer.h", "cuda__pointer_8h.html", "cuda__pointer_8h" ],
doc/html/files_dup.js:    [ "force_gpu_cuda.hpp", "force__gpu__cuda_8hpp.html", "force__gpu__cuda_8hpp" ],
doc/html/force__gpu__cuda_8hpp__incl.map:<map id="force_gpu_cuda.hpp" name="force_gpu_cuda.hpp">
doc/html/inherit_graph_11.map:<area shape="rect" id="node1" href="$structCalcForceWithLinearCutoffCUDA.html" title=" " alt="" coords="5,5,237,32"/>
doc/html/cuda__pointer_8h__incl.map:<map id="cuda_pointer.h" name="cuda_pointer.h">
doc/html/cuda__pointer_8h.js:var cuda__pointer_8h =
doc/html/cuda__pointer_8h.js:    [ "cudaPointer", "structcudaPointer.html", "structcudaPointer" ],
doc/html/cuda__pointer_8h.js:    [ "CUDA_SAFE_CALL", "cuda__pointer_8h.html#a9c46a4140aedb851be1e946d95798a6d", null ]
doc/html/changeover_8hpp__dep__incl.map:<area shape="rect" id="node14" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="525,379,677,405"/>
doc/html/force__gpu__cuda_8hpp_aedbe85337286f7c675e64a36cb78d099_icgraph.map:<map id="RetrieveForceCUDA" name="RetrieveForceCUDA">
doc/html/search/functions_1.js:  ['allocate_1095',['allocate',['../structcudaPointer.html#aa17523c0b8cf0f7931f56337a045e538',1,'cudaPointer']]],
doc/html/search/functions_6.js:  ['free_1207',['free',['../structcudaPointer.html#ade534dd16f9903eed7acbd5ab42dac63',1,'cudaPointer']]],
doc/html/search/all_5.js:  ['eps2_216',['eps2',['../structCalcForceWithLinearCutoffCUDA.html#a6fe0d42fd8bc999d02a9173e8f9fa824',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/classes_2.js:  ['calcforcewithlinearcutoffcuda_959',['CalcForceWithLinearCutoffCUDA',['../structCalcForceWithLinearCutoffCUDA.html',1,'']]],
doc/html/search/classes_2.js:  ['cudapointer_963',['cudaPointer',['../structcudaPointer.html',1,'']]]
doc/html/search/defines_9.js:  ['spjsoft_1959',['SPJSoft',['../force__gpu__cuda_8hpp.html#aaacf9eee49e4260d214d9be1f494a194',1,'SPJSoft():&#160;force_gpu_cuda.hpp'],['../simd__test_8cxx.html#aaacf9eee49e4260d214d9be1f494a194',1,'SPJSoft():&#160;simd_test.cxx']]]
doc/html/search/all_6.js:  ['force_5fgpu_5fcuda_2ehpp_262',['force_gpu_cuda.hpp',['../force__gpu__cuda_8hpp.html',1,'']]],
doc/html/search/all_6.js:  ['free_271',['free',['../structcudaPointer.html#ade534dd16f9903eed7acbd5ab42dac63',1,'cudaPointer']]],
doc/html/search/variables_11.js:  ['size_1841',['size',['../structcudaPointer.html#a29ff9caea3ff414bfa11a5284b9981ff',1,'cudaPointer::size()'],['../classHardDumpList.html#ad181cd55f0e72a7f836add7583561a6b',1,'HardDumpList::size()']]],
doc/html/search/all_7.js:  ['g_276',['G',['../structCalcForceWithLinearCutoffCUDA.html#a78de7001494564b6527792383d9c0aee',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/all_1.js:  ['allocate_18',['allocate',['../structcudaPointer.html#aa17523c0b8cf0f7931f56337a045e538',1,'cudaPointer']]],
doc/html/search/variables_10.js:  ['rcut2_1817',['rcut2',['../structCalcForceWithLinearCutoffCUDA.html#aa77b2d5759a9bceca79f46396ab0b0fc',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/all_11.js:  ['size_824',['size',['../structcudaPointer.html#a29ff9caea3ff414bfa11a5284b9981ff',1,'cudaPointer::size()'],['../classHardDumpList.html#ad181cd55f0e72a7f836add7583561a6b',1,'HardDumpList::size()']]],
doc/html/search/all_11.js:  ['spjsoft_830',['SPJSoft',['../force__gpu__cuda_8hpp.html#aaacf9eee49e4260d214d9be1f494a194',1,'SPJSoft():&#160;force_gpu_cuda.hpp'],['../simd__test_8cxx.html#aaacf9eee49e4260d214d9be1f494a194',1,'SPJSoft():&#160;simd_test.cxx']]],
doc/html/search/all_8.js:  ['host_5fpointer_425',['host_pointer',['../structcudaPointer.html#a779bb7d0805ec2fe5682263c913b6b35',1,'cudaPointer']]],
doc/html/search/all_8.js:  ['htod_426',['htod',['../structcudaPointer.html#afe2d54bb2bc70a581bc095d2977be1cf',1,'cudaPointer::htod(int count)'],['../structcudaPointer.html#a6706a86ab06a072c74866c3786526dcc',1,'cudaPointer::htod()']]]
doc/html/search/functions_8.js:  ['htod_1312',['htod',['../structcudaPointer.html#afe2d54bb2bc70a581bc095d2977be1cf',1,'cudaPointer::htod(int count)'],['../structcudaPointer.html#a6706a86ab06a072c74866c3786526dcc',1,'cudaPointer::htod()']]]
doc/html/search/all_4.js:  ['dev_5fpointer_165',['dev_pointer',['../structcudaPointer.html#a333bb3eb7d55306bfc90090f801d38f6',1,'cudaPointer']]],
doc/html/search/all_4.js:  ['dtoh_186',['dtoh',['../structcudaPointer.html#ac2a111234d85904b3a3d5030e8c19f12',1,'cudaPointer::dtoh(int count)'],['../structcudaPointer.html#a2b3f425013c37af90186957e79ed9285',1,'cudaPointer::dtoh()']]],
doc/html/search/variables_8.js:  ['host_5fpointer_1651',['host_pointer',['../structcudaPointer.html#a779bb7d0805ec2fe5682263c913b6b35',1,'cudaPointer']]]
doc/html/search/variables_7.js:  ['g_1625',['G',['../structCalcForceWithLinearCutoffCUDA.html#a78de7001494564b6527792383d9c0aee',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/files_2.js:  ['cuda_5fpointer_2eh_1043',['cuda_pointer.h',['../cuda__pointer_8h.html',1,'']]]
doc/html/search/variables_4.js:  ['dev_5fpointer_1575',['dev_pointer',['../structcudaPointer.html#a333bb3eb7d55306bfc90090f801d38f6',1,'cudaPointer']]],
doc/html/search/variables_5.js:  ['eps2_1604',['eps2',['../structCalcForceWithLinearCutoffCUDA.html#a6fe0d42fd8bc999d02a9173e8f9fa824',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/functions_9.js:  ['initialize_1315',['initialize',['../classSearchCluster.html#a80b0896173dc501cd2457c1636289206',1,'SearchCluster::initialize()'],['../structCalcForceWithLinearCutoffCUDA.html#a6f205d494df72fdecaa0919072b1a029',1,'CalcForceWithLinearCutoffCUDA::initialize()']]],
doc/html/search/functions_10.js:  ['retrieveforcecuda_1419',['RetrieveForceCUDA',['../force__gpu__cuda_8hpp.html#aedbe85337286f7c675e64a36cb78d099',1,'force_gpu_cuda.hpp']]],
doc/html/search/functions_e.js:  ['operator_20t_2a_1355',['operator T*',['../structcudaPointer.html#a6d4a4b40830158023a35071274099943',1,'cudaPointer']]],
doc/html/search/functions_e.js:  ['operator_28_29_1360',['operator()',['../structCalcForceWithLinearCutoffCUDA.html#a87f439a8341d16b91764f0483d326a89',1,'CalcForceWithLinearCutoffCUDA::operator()()'],['../structSearchNeighborEpEpNoSimd.html#ad642918192c50ebb872db6c913da12ce',1,'SearchNeighborEpEpNoSimd::operator()()'],['../structCalcForceEpEpWithLinearCutoffNoSimd.html#a484f402f8f1504ea925a1aaa5974af3d',1,'CalcForceEpEpWithLinearCutoffNoSimd::operator()()'],['../structCalcForceEpSpMonoNoSimd.html#af00e4f24efe9c66ceee99edc1324159e',1,'CalcForceEpSpMonoNoSimd::operator()()'],['../structCalcForceEpSpQuadNoSimd.html#a812fee500cefba94ab6ad6300eef9199',1,'CalcForceEpSpQuadNoSimd::operator()()'],['../structCalcForcePPNoSimd.html#a07e2c60d4eb540bb73d207edbee4068f',1,'CalcForcePPNoSimd::operator()()']]],
doc/html/search/functions_e.js:  ['operator_5b_5d_1373',['operator[]',['../structcudaPointer.html#a657565d2687c0db348631fd7679a95d9',1,'cudaPointer::operator[]()'],['../classHardDumpList.html#a4a6d1fd1e29020d55ee9ac9923881200',1,'HardDumpList::operator[]()'],['../classPIKG_1_1Vector3.html#a651558af7c797690c04447bbcce07af6',1,'PIKG::Vector3::operator[](const int i) const'],['../classPIKG_1_1Vector3.html#a7c3a8d0a2cc7e77d7dd309df6957c73f',1,'PIKG::Vector3::operator[](const int i)'],['../classPIKG_1_1Vector2.html#afd347dff84add8b559130fc015ae2f80',1,'PIKG::Vector2::operator[](const int i)'],['../classPIKG_1_1Vector2.html#a9a211516d8217a6526877b845e95c06c',1,'PIKG::Vector2::operator[](const int i) const'],['../classPIKG_1_1Vector4.html#a6c986fb9d6fdf334ed69f8b78d112500',1,'PIKG::Vector4::operator[](const int i) const'],['../classPIKG_1_1Vector4.html#ac8f3b4563d0a0eca438c12f52c930795',1,'PIKG::Vector4::operator[](const int i)']]],
doc/html/search/files_5.js:  ['force_5fgpu_5fcuda_2ehpp_1049',['force_gpu_cuda.hpp',['../force__gpu__cuda_8hpp.html',1,'']]],
doc/html/search/functions_4.js:  ['dtoh_1177',['dtoh',['../structcudaPointer.html#ac2a111234d85904b3a3d5030e8c19f12',1,'cudaPointer::dtoh(int count)'],['../structcudaPointer.html#a2b3f425013c37af90186957e79ed9285',1,'cudaPointer::dtoh()']]],
doc/html/search/all_3.js:  ['calcforcewithlinearcutoffcuda_89',['CalcForceWithLinearCutoffCUDA',['../structCalcForceWithLinearCutoffCUDA.html',1,'CalcForceWithLinearCutoffCUDA'],['../structCalcForceWithLinearCutoffCUDA.html#a66c7b35fafd0f18545f3e29e45e47036',1,'CalcForceWithLinearCutoffCUDA::CalcForceWithLinearCutoffCUDA()'],['../structCalcForceWithLinearCutoffCUDA.html#abaa7f5bafe1d976584fcd89797ab21fd',1,'CalcForceWithLinearCutoffCUDA::CalcForceWithLinearCutoffCUDA(PS::S32 _rank, PS::F64 _eps2, PS::F64 _rcut2, PS::F64 _G)']]],
doc/html/search/all_3.js:  ['cuda_5fpointer_2eh_146',['cuda_pointer.h',['../cuda__pointer_8h.html',1,'']]],
doc/html/search/all_3.js:  ['cuda_5fsafe_5fcall_147',['CUDA_SAFE_CALL',['../cuda__pointer_8h.html#a9c46a4140aedb851be1e946d95798a6d',1,'cuda_pointer.h']]],
doc/html/search/all_3.js:  ['cudapointer_148',['cudaPointer',['../structcudaPointer.html',1,'cudaPointer&lt; T &gt;'],['../structcudaPointer.html#af3030840ad5ab7e0ba1f490f567c50d9',1,'cudaPointer::cudaPointer()']]]
doc/html/search/functions_3.js:  ['calcforcewithlinearcutoffcuda_1127',['CalcForceWithLinearCutoffCUDA',['../structCalcForceWithLinearCutoffCUDA.html#a66c7b35fafd0f18545f3e29e45e47036',1,'CalcForceWithLinearCutoffCUDA::CalcForceWithLinearCutoffCUDA()'],['../structCalcForceWithLinearCutoffCUDA.html#abaa7f5bafe1d976584fcd89797ab21fd',1,'CalcForceWithLinearCutoffCUDA::CalcForceWithLinearCutoffCUDA(PS::S32 _rank, PS::F64 _eps2, PS::F64 _rcut2, PS::F64 _G)']]],
doc/html/search/functions_3.js:  ['cudapointer_1170',['cudaPointer',['../structcudaPointer.html#af3030840ad5ab7e0ba1f490f567c50d9',1,'cudaPointer']]]
doc/html/search/all_c.js:  ['my_5frank_523',['my_rank',['../structCalcForceWithLinearCutoffCUDA.html#a534005d7cd4e0de5fa39def30df85be0',1,'CalcForceWithLinearCutoffCUDA']]]
doc/html/search/all_9.js:  ['initialize_438',['initialize',['../classSearchCluster.html#a80b0896173dc501cd2457c1636289206',1,'SearchCluster::initialize()'],['../structCalcForceWithLinearCutoffCUDA.html#a6f205d494df72fdecaa0919072b1a029',1,'CalcForceWithLinearCutoffCUDA::initialize()']]],
doc/html/search/all_e.js:  ['operator_20t_2a_587',['operator T*',['../structcudaPointer.html#a6d4a4b40830158023a35071274099943',1,'cudaPointer']]],
doc/html/search/all_e.js:  ['operator_28_29_592',['operator()',['../structCalcForceWithLinearCutoffCUDA.html#a87f439a8341d16b91764f0483d326a89',1,'CalcForceWithLinearCutoffCUDA::operator()()'],['../structSearchNeighborEpEpNoSimd.html#ad642918192c50ebb872db6c913da12ce',1,'SearchNeighborEpEpNoSimd::operator()()'],['../structCalcForceEpEpWithLinearCutoffNoSimd.html#a484f402f8f1504ea925a1aaa5974af3d',1,'CalcForceEpEpWithLinearCutoffNoSimd::operator()()'],['../structCalcForceEpSpMonoNoSimd.html#af00e4f24efe9c66ceee99edc1324159e',1,'CalcForceEpSpMonoNoSimd::operator()()'],['../structCalcForceEpSpQuadNoSimd.html#a812fee500cefba94ab6ad6300eef9199',1,'CalcForceEpSpQuadNoSimd::operator()()'],['../structCalcForcePPNoSimd.html#a07e2c60d4eb540bb73d207edbee4068f',1,'CalcForcePPNoSimd::operator()()']]],
doc/html/search/all_e.js:  ['operator_5b_5d_606',['operator[]',['../structcudaPointer.html#a657565d2687c0db348631fd7679a95d9',1,'cudaPointer::operator[]()'],['../classHardDumpList.html#a4a6d1fd1e29020d55ee9ac9923881200',1,'HardDumpList::operator[]()'],['../classPIKG_1_1Vector3.html#a651558af7c797690c04447bbcce07af6',1,'PIKG::Vector3::operator[](const int i) const'],['../classPIKG_1_1Vector3.html#a7c3a8d0a2cc7e77d7dd309df6957c73f',1,'PIKG::Vector3::operator[](const int i)'],['../classPIKG_1_1Vector2.html#afd347dff84add8b559130fc015ae2f80',1,'PIKG::Vector2::operator[](const int i)'],['../classPIKG_1_1Vector2.html#a9a211516d8217a6526877b845e95c06c',1,'PIKG::Vector2::operator[](const int i) const'],['../classPIKG_1_1Vector4.html#a6c986fb9d6fdf334ed69f8b78d112500',1,'PIKG::Vector4::operator[](const int i) const'],['../classPIKG_1_1Vector4.html#ac8f3b4563d0a0eca438c12f52c930795',1,'PIKG::Vector4::operator[](const int i)']]],
doc/html/search/variables_c.js:  ['my_5frank_1697',['my_rank',['../structCalcForceWithLinearCutoffCUDA.html#a534005d7cd4e0de5fa39def30df85be0',1,'CalcForceWithLinearCutoffCUDA']]]
doc/html/search/all_10.js:  ['rcut2_724',['rcut2',['../structCalcForceWithLinearCutoffCUDA.html#aa77b2d5759a9bceca79f46396ab0b0fc',1,'CalcForceWithLinearCutoffCUDA']]],
doc/html/search/all_10.js:  ['retrieveforcecuda_743',['RetrieveForceCUDA',['../force__gpu__cuda_8hpp.html#aedbe85337286f7c675e64a36cb78d099',1,'force_gpu_cuda.hpp']]],
doc/html/search/defines_3.js:  ['cuda_5fsafe_5fcall_1951',['CUDA_SAFE_CALL',['../cuda__pointer_8h.html#a9c46a4140aedb851be1e946d95798a6d',1,'cuda_pointer.h']]]
doc/html/annotated_dup.js:    [ "CalcForceWithLinearCutoffCUDA", "structCalcForceWithLinearCutoffCUDA.html", "structCalcForceWithLinearCutoffCUDA" ],
doc/html/annotated_dup.js:    [ "cudaPointer", "structcudaPointer.html", "structcudaPointer" ],
doc/html/navtreeindex6.js:"structcudaPointer.html":[2,0,16],
doc/html/navtreeindex6.js:"structcudaPointer.html#a29ff9caea3ff414bfa11a5284b9981ff":[2,0,16,11],
doc/html/navtreeindex6.js:"structcudaPointer.html#a2b3f425013c37af90186957e79ed9285":[2,0,16,2],
doc/html/navtreeindex6.js:"structcudaPointer.html#a333bb3eb7d55306bfc90090f801d38f6":[2,0,16,9],
doc/html/navtreeindex6.js:"structcudaPointer.html#a657565d2687c0db348631fd7679a95d9":[2,0,16,8],
doc/html/navtreeindex6.js:"structcudaPointer.html#a6706a86ab06a072c74866c3786526dcc":[2,0,16,5],
doc/html/navtreeindex6.js:"structcudaPointer.html#a6d4a4b40830158023a35071274099943":[2,0,16,7],
doc/html/navtreeindex6.js:"structcudaPointer.html#a779bb7d0805ec2fe5682263c913b6b35":[2,0,16,10],
doc/html/navtreeindex6.js:"structcudaPointer.html#aa17523c0b8cf0f7931f56337a045e538":[2,0,16,1],
doc/html/navtreeindex6.js:"structcudaPointer.html#ac2a111234d85904b3a3d5030e8c19f12":[2,0,16,3],
doc/html/navtreeindex6.js:"structcudaPointer.html#ade534dd16f9903eed7acbd5ab42dac63":[2,0,16,4],
doc/html/navtreeindex6.js:"structcudaPointer.html#af3030840ad5ab7e0ba1f490f567c50d9":[2,0,16,0],
doc/html/navtreeindex6.js:"structcudaPointer.html#afe2d54bb2bc70a581bc095d2977be1cf":[2,0,16,6],
doc/html/ptcl_8hpp__dep__incl.map:<area shape="rect" id="node13" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="353,304,505,331"/>
doc/html/hierarchy.js:    [ "CalcForceWithLinearCutoffCUDA", "structCalcForceWithLinearCutoffCUDA.html", null ],
doc/html/hierarchy.js:    [ "cudaPointer< T >", "structcudaPointer.html", null ],
doc/html/pseudoparticle__multipole_8hpp__dep__incl.map:<area shape="rect" id="node15" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="170,453,322,480"/>
doc/html/structCalcForceWithLinearCutoffCUDA.js:var structCalcForceWithLinearCutoffCUDA =
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "CalcForceWithLinearCutoffCUDA", "structCalcForceWithLinearCutoffCUDA.html#a66c7b35fafd0f18545f3e29e45e47036", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "CalcForceWithLinearCutoffCUDA", "structCalcForceWithLinearCutoffCUDA.html#abaa7f5bafe1d976584fcd89797ab21fd", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "initialize", "structCalcForceWithLinearCutoffCUDA.html#a6f205d494df72fdecaa0919072b1a029", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "operator()", "structCalcForceWithLinearCutoffCUDA.html#a87f439a8341d16b91764f0483d326a89", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "eps2", "structCalcForceWithLinearCutoffCUDA.html#a6fe0d42fd8bc999d02a9173e8f9fa824", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "G", "structCalcForceWithLinearCutoffCUDA.html#a78de7001494564b6527792383d9c0aee", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "my_rank", "structCalcForceWithLinearCutoffCUDA.html#a534005d7cd4e0de5fa39def30df85be0", null ],
doc/html/structCalcForceWithLinearCutoffCUDA.js:    [ "rcut2", "structCalcForceWithLinearCutoffCUDA.html#aa77b2d5759a9bceca79f46396ab0b0fc", null ]
doc/html/navtreeindex4.js:"cuda__pointer_8h.html":[3,0,7],
doc/html/navtreeindex4.js:"cuda__pointer_8h.html#a9c46a4140aedb851be1e946d95798a6d":[3,0,7,1],
doc/html/navtreeindex4.js:"cuda__pointer_8h_source.html":[3,0,7],
doc/html/navtreeindex4.js:"force__gpu__cuda_8hpp.html":[3,0,13],
doc/html/navtreeindex4.js:"force__gpu__cuda_8hpp.html#aaacf9eee49e4260d214d9be1f494a194":[3,0,13,1],
doc/html/navtreeindex4.js:"force__gpu__cuda_8hpp.html#aedbe85337286f7c675e64a36cb78d099":[3,0,13,2],
doc/html/navtreeindex4.js:"force__gpu__cuda_8hpp_source.html":[3,0,13],
doc/html/tidal__tensor_8hpp__dep__incl.map:<area shape="rect" id="node15" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="507,453,659,480"/>
doc/html/soft__ptcl_8hpp__dep__incl.map:<area shape="rect" id="node2" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="5,80,157,107"/>
doc/html/structcudaPointer.js:var structcudaPointer =
doc/html/structcudaPointer.js:    [ "cudaPointer", "structcudaPointer.html#af3030840ad5ab7e0ba1f490f567c50d9", null ],
doc/html/structcudaPointer.js:    [ "allocate", "structcudaPointer.html#aa17523c0b8cf0f7931f56337a045e538", null ],
doc/html/structcudaPointer.js:    [ "dtoh", "structcudaPointer.html#a2b3f425013c37af90186957e79ed9285", null ],
doc/html/structcudaPointer.js:    [ "dtoh", "structcudaPointer.html#ac2a111234d85904b3a3d5030e8c19f12", null ],
doc/html/structcudaPointer.js:    [ "free", "structcudaPointer.html#ade534dd16f9903eed7acbd5ab42dac63", null ],
doc/html/structcudaPointer.js:    [ "htod", "structcudaPointer.html#a6706a86ab06a072c74866c3786526dcc", null ],
doc/html/structcudaPointer.js:    [ "htod", "structcudaPointer.html#afe2d54bb2bc70a581bc095d2977be1cf", null ],
doc/html/structcudaPointer.js:    [ "operator T*", "structcudaPointer.html#a6d4a4b40830158023a35071274099943", null ],
doc/html/structcudaPointer.js:    [ "operator[]", "structcudaPointer.html#a657565d2687c0db348631fd7679a95d9", null ],
doc/html/structcudaPointer.js:    [ "dev_pointer", "structcudaPointer.html#a333bb3eb7d55306bfc90090f801d38f6", null ],
doc/html/structcudaPointer.js:    [ "host_pointer", "structcudaPointer.html#a779bb7d0805ec2fe5682263c913b6b35", null ],
doc/html/structcudaPointer.js:    [ "size", "structcudaPointer.html#a29ff9caea3ff414bfa11a5284b9981ff", null ]
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html":[2,0,12],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a534005d7cd4e0de5fa39def30df85be0":[2,0,12,6],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a66c7b35fafd0f18545f3e29e45e47036":[2,0,12,0],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a6f205d494df72fdecaa0919072b1a029":[2,0,12,2],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a6fe0d42fd8bc999d02a9173e8f9fa824":[2,0,12,4],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a78de7001494564b6527792383d9c0aee":[2,0,12,5],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#a87f439a8341d16b91764f0483d326a89":[2,0,12,3],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#aa77b2d5759a9bceca79f46396ab0b0fc":[2,0,12,7],
doc/html/navtreeindex5.js:"structCalcForceWithLinearCutoffCUDA.html#abaa7f5bafe1d976584fcd89797ab21fd":[2,0,12,1],
doc/html/particle__base_8hpp__dep__incl.map:<area shape="rect" id="node14" href="$force__gpu__cuda_8hpp.html" title=" " alt="" coords="457,379,609,405"/>
doc/latex/hierarchy.tex:\item \contentsline{section}{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA}{\pageref{structCalcForceWithLinearCutoffCUDA}}{}
doc/latex/hierarchy.tex:\item \contentsline{section}{cuda\+Pointer$<$ T $>$}{\pageref{structcudaPointer}}{}
doc/latex/cuda__pointer_8h.tex:\hypertarget{cuda__pointer_8h}{}\doxysection{cuda\+\_\+pointer.\+h File Reference}
doc/latex/cuda__pointer_8h.tex:\label{cuda__pointer_8h}\index{cuda\_pointer.h@{cuda\_pointer.h}}
doc/latex/cuda__pointer_8h.tex:{\ttfamily \#include $<$cuda.\+h$>$}\newline
doc/latex/cuda__pointer_8h.tex:{\ttfamily \#include $<$cuda\+\_\+runtime.\+h$>$}\newline
doc/latex/cuda__pointer_8h.tex:Include dependency graph for cuda\+\_\+pointer.\+h\+:\nopagebreak
doc/latex/cuda__pointer_8h.tex:\includegraphics[width=314pt]{cuda__pointer_8h__incl}
doc/latex/cuda__pointer_8h.tex:struct \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer$<$ T $>$}}
doc/latex/cuda__pointer_8h.tex:\#define \mbox{\hyperlink{cuda__pointer_8h_a9c46a4140aedb851be1e946d95798a6d}{C\+U\+D\+A\+\_\+\+S\+A\+F\+E\+\_\+\+C\+A\+LL}}(val)~val
doc/latex/cuda__pointer_8h.tex:\mbox{\Hypertarget{cuda__pointer_8h_a9c46a4140aedb851be1e946d95798a6d}\label{cuda__pointer_8h_a9c46a4140aedb851be1e946d95798a6d}} 
doc/latex/cuda__pointer_8h.tex:\index{cuda\_pointer.h@{cuda\_pointer.h}!CUDA\_SAFE\_CALL@{CUDA\_SAFE\_CALL}}
doc/latex/cuda__pointer_8h.tex:\index{CUDA\_SAFE\_CALL@{CUDA\_SAFE\_CALL}!cuda\_pointer.h@{cuda\_pointer.h}}
doc/latex/cuda__pointer_8h.tex:\doxysubsubsection{\texorpdfstring{CUDA\_SAFE\_CALL}{CUDA\_SAFE\_CALL}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\hypertarget{structCalcForceWithLinearCutoffCUDA}{}\doxysection{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA Struct Reference}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\label{structCalcForceWithLinearCutoffCUDA}\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:{\ttfamily \#include $<$force\+\_\+gpu\+\_\+cuda.\+hpp$>$}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a66c7b35fafd0f18545f3e29e45e47036}{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA}} ()
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_abaa7f5bafe1d976584fcd89797ab21fd}{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA}} (P\+S\+::\+S32 \+\_\+rank, P\+S\+::\+F64 \+\_\+eps2, P\+S\+::\+F64 \+\_\+rcut2, P\+S\+::\+F64 \+\_\+G)
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:void \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a6f205d494df72fdecaa0919072b1a029}{initialize}} (P\+S\+::\+S32 \+\_\+rank, P\+S\+::\+F64 \+\_\+eps2, P\+S\+::\+F64 \+\_\+rcut2, P\+S\+::\+F64 \+\_\+G)
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:P\+S\+::\+S32 \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a87f439a8341d16b91764f0483d326a89}{operator()}} (const P\+S\+::\+S32 tag, const P\+S\+::\+S32 n\+\_\+walk, const \mbox{\hyperlink{classEPISoft}{E\+P\+I\+Soft}} $\ast$epi\mbox{[}$\,$\mbox{]}, const P\+S\+::\+S32 n\+\_\+epi\mbox{[}$\,$\mbox{]}, const \mbox{\hyperlink{classEPJSoft}{E\+P\+J\+Soft}} $\ast$epj\mbox{[}$\,$\mbox{]}, const P\+S\+::\+S32 n\+\_\+epj\mbox{[}$\,$\mbox{]}, const \mbox{\hyperlink{simd__test_8cxx_aaacf9eee49e4260d214d9be1f494a194}{S\+P\+J\+Soft}} $\ast$spj\mbox{[}$\,$\mbox{]}, const P\+S\+::\+S32 n\+\_\+spj\mbox{[}$\,$\mbox{]})
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:P\+S\+::\+S32 \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a534005d7cd4e0de5fa39def30df85be0}{my\+\_\+rank}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:P\+S\+::\+F64 \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a6fe0d42fd8bc999d02a9173e8f9fa824}{eps2}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:P\+S\+::\+F64 \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_aa77b2d5759a9bceca79f46396ab0b0fc}{rcut2}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:P\+S\+::\+F64 \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA_a78de7001494564b6527792383d9c0aee}{G}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a66c7b35fafd0f18545f3e29e45e47036}\label{structCalcForceWithLinearCutoffCUDA_a66c7b35fafd0f18545f3e29e45e47036}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\doxysubsubsection{\texorpdfstring{CalcForceWithLinearCutoffCUDA()}{CalcForceWithLinearCutoffCUDA()}\hspace{0.1cm}{\footnotesize\ttfamily [1/2]}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_abaa7f5bafe1d976584fcd89797ab21fd}\label{structCalcForceWithLinearCutoffCUDA_abaa7f5bafe1d976584fcd89797ab21fd}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\doxysubsubsection{\texorpdfstring{CalcForceWithLinearCutoffCUDA()}{CalcForceWithLinearCutoffCUDA()}\hspace{0.1cm}{\footnotesize\ttfamily [2/2]}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a6f205d494df72fdecaa0919072b1a029}\label{structCalcForceWithLinearCutoffCUDA_a6f205d494df72fdecaa0919072b1a029}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!initialize@{initialize}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{initialize@{initialize}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a87f439a8341d16b91764f0483d326a89}\label{structCalcForceWithLinearCutoffCUDA_a87f439a8341d16b91764f0483d326a89}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!operator()@{operator()}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{operator()@{operator()}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a6fe0d42fd8bc999d02a9173e8f9fa824}\label{structCalcForceWithLinearCutoffCUDA_a6fe0d42fd8bc999d02a9173e8f9fa824}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!eps2@{eps2}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{eps2@{eps2}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a78de7001494564b6527792383d9c0aee}\label{structCalcForceWithLinearCutoffCUDA_a78de7001494564b6527792383d9c0aee}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!G@{G}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{G@{G}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_a534005d7cd4e0de5fa39def30df85be0}\label{structCalcForceWithLinearCutoffCUDA_a534005d7cd4e0de5fa39def30df85be0}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!my\_rank@{my\_rank}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{my\_rank@{my\_rank}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\Hypertarget{structCalcForceWithLinearCutoffCUDA_aa77b2d5759a9bceca79f46396ab0b0fc}\label{structCalcForceWithLinearCutoffCUDA_aa77b2d5759a9bceca79f46396ab0b0fc}} 
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}!rcut2@{rcut2}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\index{rcut2@{rcut2}!CalcForceWithLinearCutoffCUDA@{CalcForceWithLinearCutoffCUDA}}
doc/latex/structCalcForceWithLinearCutoffCUDA.tex:\mbox{\hyperlink{force__gpu__cuda_8hpp}{force\+\_\+gpu\+\_\+cuda.\+hpp}}\end{DoxyCompactItemize}
doc/latex/annotated.tex:\item\contentsline{section}{\mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA}{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA}} }{\pageref{structCalcForceWithLinearCutoffCUDA}}{}
doc/latex/annotated.tex:\item\contentsline{section}{\mbox{\hyperlink{structcudaPointer}{cuda\+Pointer$<$ T $>$}} }{\pageref{structcudaPointer}}{}
doc/latex/structcudaPointer.tex:\hypertarget{structcudaPointer}{}\doxysection{cuda\+Pointer$<$ T $>$ Struct Template Reference}
doc/latex/structcudaPointer.tex:\label{structcudaPointer}\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:{\ttfamily \#include $<$cuda\+\_\+pointer.\+h$>$}
doc/latex/structcudaPointer.tex:\mbox{\hyperlink{structcudaPointer_af3030840ad5ab7e0ba1f490f567c50d9}{cuda\+Pointer}} ()
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_aa17523c0b8cf0f7931f56337a045e538}{allocate}} (int \+\_\+size)
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_ade534dd16f9903eed7acbd5ab42dac63}{free}} ()
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_afe2d54bb2bc70a581bc095d2977be1cf}{htod}} (int count)
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_a6706a86ab06a072c74866c3786526dcc}{htod}} ()
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_ac2a111234d85904b3a3d5030e8c19f12}{dtoh}} (int count)
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer_a2b3f425013c37af90186957e79ed9285}{dtoh}} ()
doc/latex/structcudaPointer.tex:T \& \mbox{\hyperlink{structcudaPointer_a657565d2687c0db348631fd7679a95d9}{operator\mbox{[}$\,$\mbox{]}}} (int i)
doc/latex/structcudaPointer.tex:\mbox{\hyperlink{structcudaPointer_a6d4a4b40830158023a35071274099943}{operator T$\ast$}} ()
doc/latex/structcudaPointer.tex:T $\ast$ \mbox{\hyperlink{structcudaPointer_a333bb3eb7d55306bfc90090f801d38f6}{dev\+\_\+pointer}}
doc/latex/structcudaPointer.tex:T $\ast$ \mbox{\hyperlink{structcudaPointer_a779bb7d0805ec2fe5682263c913b6b35}{host\+\_\+pointer}}
doc/latex/structcudaPointer.tex:size\+\_\+t \mbox{\hyperlink{structcudaPointer_a29ff9caea3ff414bfa11a5284b9981ff}{size}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_af3030840ad5ab7e0ba1f490f567c50d9}\label{structcudaPointer_af3030840ad5ab7e0ba1f490f567c50d9}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!cudaPointer@{cudaPointer}}
doc/latex/structcudaPointer.tex:\index{cudaPointer@{cudaPointer}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:\doxysubsubsection{\texorpdfstring{cudaPointer()}{cudaPointer()}}
doc/latex/structcudaPointer.tex:\mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::\mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}} (\begin{DoxyParamCaption}{ }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_aa17523c0b8cf0f7931f56337a045e538}\label{structcudaPointer_aa17523c0b8cf0f7931f56337a045e538}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!allocate@{allocate}}
doc/latex/structcudaPointer.tex:\index{allocate@{allocate}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::allocate (\begin{DoxyParamCaption}\item[{int}]{\+\_\+size }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a2b3f425013c37af90186957e79ed9285}\label{structcudaPointer_a2b3f425013c37af90186957e79ed9285}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!dtoh@{dtoh}}
doc/latex/structcudaPointer.tex:\index{dtoh@{dtoh}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::dtoh (\begin{DoxyParamCaption}{ }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_ac2a111234d85904b3a3d5030e8c19f12}\label{structcudaPointer_ac2a111234d85904b3a3d5030e8c19f12}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!dtoh@{dtoh}}
doc/latex/structcudaPointer.tex:\index{dtoh@{dtoh}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::dtoh (\begin{DoxyParamCaption}\item[{int}]{count }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_ade534dd16f9903eed7acbd5ab42dac63}\label{structcudaPointer_ade534dd16f9903eed7acbd5ab42dac63}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!free@{free}}
doc/latex/structcudaPointer.tex:\index{free@{free}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::free (\begin{DoxyParamCaption}{ }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a6706a86ab06a072c74866c3786526dcc}\label{structcudaPointer_a6706a86ab06a072c74866c3786526dcc}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!htod@{htod}}
doc/latex/structcudaPointer.tex:\index{htod@{htod}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::htod (\begin{DoxyParamCaption}{ }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_afe2d54bb2bc70a581bc095d2977be1cf}\label{structcudaPointer_afe2d54bb2bc70a581bc095d2977be1cf}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!htod@{htod}}
doc/latex/structcudaPointer.tex:\index{htod@{htod}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:void \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::htod (\begin{DoxyParamCaption}\item[{int}]{count }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a6d4a4b40830158023a35071274099943}\label{structcudaPointer_a6d4a4b40830158023a35071274099943}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!operator T$\ast$@{operator T$\ast$}}
doc/latex/structcudaPointer.tex:\index{operator T$\ast$@{operator T$\ast$}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:\mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::operator T$\ast$ (\begin{DoxyParamCaption}{ }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a657565d2687c0db348631fd7679a95d9}\label{structcudaPointer_a657565d2687c0db348631fd7679a95d9}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!operator\mbox{[}\mbox{]}@{operator[]}}
doc/latex/structcudaPointer.tex:\index{operator\mbox{[}\mbox{]}@{operator[]}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:T\& \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::operator\mbox{[}$\,$\mbox{]} (\begin{DoxyParamCaption}\item[{int}]{i }\end{DoxyParamCaption})\hspace{0.3cm}{\ttfamily [inline]}}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a333bb3eb7d55306bfc90090f801d38f6}\label{structcudaPointer_a333bb3eb7d55306bfc90090f801d38f6}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!dev\_pointer@{dev\_pointer}}
doc/latex/structcudaPointer.tex:\index{dev\_pointer@{dev\_pointer}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:T$\ast$ \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::dev\+\_\+pointer}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a779bb7d0805ec2fe5682263c913b6b35}\label{structcudaPointer_a779bb7d0805ec2fe5682263c913b6b35}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!host\_pointer@{host\_pointer}}
doc/latex/structcudaPointer.tex:\index{host\_pointer@{host\_pointer}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:T$\ast$ \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::host\+\_\+pointer}
doc/latex/structcudaPointer.tex:\mbox{\Hypertarget{structcudaPointer_a29ff9caea3ff414bfa11a5284b9981ff}\label{structcudaPointer_a29ff9caea3ff414bfa11a5284b9981ff}} 
doc/latex/structcudaPointer.tex:\index{cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}!size@{size}}
doc/latex/structcudaPointer.tex:\index{size@{size}!cudaPointer$<$ T $>$@{cudaPointer$<$ T $>$}}
doc/latex/structcudaPointer.tex:size\+\_\+t \mbox{\hyperlink{structcudaPointer}{cuda\+Pointer}}$<$ T $>$\+::size}
doc/latex/structcudaPointer.tex:\mbox{\hyperlink{cuda__pointer_8h}{cuda\+\_\+pointer.\+h}}\end{DoxyCompactItemize}
doc/latex/refman.tex:\input{structCalcForceWithLinearCutoffCUDA}
doc/latex/refman.tex:\input{structcudaPointer}
doc/latex/refman.tex:\input{cuda__pointer_8h}
doc/latex/refman.tex:\input{force__gpu__cuda_8hpp}
doc/latex/force__gpu__cuda_8hpp.tex:\hypertarget{force__gpu__cuda_8hpp}{}\doxysection{force\+\_\+gpu\+\_\+cuda.\+hpp File Reference}
doc/latex/force__gpu__cuda_8hpp.tex:\label{force__gpu__cuda_8hpp}\index{force\_gpu\_cuda.hpp@{force\_gpu\_cuda.hpp}}
doc/latex/force__gpu__cuda_8hpp.tex:Include dependency graph for force\+\_\+gpu\+\_\+cuda.\+hpp\+:\nopagebreak
doc/latex/force__gpu__cuda_8hpp.tex:\includegraphics[width=350pt]{force__gpu__cuda_8hpp__incl}
doc/latex/force__gpu__cuda_8hpp.tex:struct \mbox{\hyperlink{structCalcForceWithLinearCutoffCUDA}{Calc\+Force\+With\+Linear\+Cutoff\+C\+U\+DA}}
doc/latex/force__gpu__cuda_8hpp.tex:\#define \mbox{\hyperlink{force__gpu__cuda_8hpp_aaacf9eee49e4260d214d9be1f494a194}{S\+P\+J\+Soft}}~P\+S\+::\+S\+P\+J\+Monopole\+In\+And\+Out
doc/latex/force__gpu__cuda_8hpp.tex:P\+S\+::\+S32 \mbox{\hyperlink{force__gpu__cuda_8hpp_aedbe85337286f7c675e64a36cb78d099}{Retrieve\+Force\+C\+U\+DA}} (const P\+S\+::\+S32 tag, const P\+S\+::\+S32 n\+\_\+walk, const P\+S\+::\+S32 $\ast$ni, \mbox{\hyperlink{classForceSoft}{Force\+Soft}} $\ast$$\ast$force)
doc/latex/force__gpu__cuda_8hpp.tex:\mbox{\Hypertarget{force__gpu__cuda_8hpp_aaacf9eee49e4260d214d9be1f494a194}\label{force__gpu__cuda_8hpp_aaacf9eee49e4260d214d9be1f494a194}} 
doc/latex/force__gpu__cuda_8hpp.tex:\index{force\_gpu\_cuda.hpp@{force\_gpu\_cuda.hpp}!SPJSoft@{SPJSoft}}
doc/latex/force__gpu__cuda_8hpp.tex:\index{SPJSoft@{SPJSoft}!force\_gpu\_cuda.hpp@{force\_gpu\_cuda.hpp}}
doc/latex/force__gpu__cuda_8hpp.tex:\mbox{\Hypertarget{force__gpu__cuda_8hpp_aedbe85337286f7c675e64a36cb78d099}\label{force__gpu__cuda_8hpp_aedbe85337286f7c675e64a36cb78d099}} 
doc/latex/force__gpu__cuda_8hpp.tex:\index{force\_gpu\_cuda.hpp@{force\_gpu\_cuda.hpp}!RetrieveForceCUDA@{RetrieveForceCUDA}}
doc/latex/force__gpu__cuda_8hpp.tex:\index{RetrieveForceCUDA@{RetrieveForceCUDA}!force\_gpu\_cuda.hpp@{force\_gpu\_cuda.hpp}}
doc/latex/force__gpu__cuda_8hpp.tex:\doxysubsubsection{\texorpdfstring{RetrieveForceCUDA()}{RetrieveForceCUDA()}}
doc/latex/force__gpu__cuda_8hpp.tex:\includegraphics[width=270pt]{force__gpu__cuda_8hpp_aedbe85337286f7c675e64a36cb78d099_icgraph}
doc/latex/dir_68267d1309a1af8e8297ef4c3efbcdba.tex:file \mbox{\hyperlink{cuda__pointer_8h}{cuda\+\_\+pointer.\+h}}
doc/latex/dir_68267d1309a1af8e8297ef4c3efbcdba.tex:file \mbox{\hyperlink{force__gpu__cuda_8hpp}{force\+\_\+gpu\+\_\+cuda.\+hpp}}
doc/latex/files.tex:\item\contentsline{section}{\mbox{\hyperlink{cuda__pointer_8h}{cuda\+\_\+pointer.\+h}} }{\pageref{cuda__pointer_8h}}{}
doc/latex/files.tex:\item\contentsline{section}{\mbox{\hyperlink{force__gpu__cuda_8hpp}{force\+\_\+gpu\+\_\+cuda.\+hpp}} }{\pageref{force__gpu__cuda_8hpp}}{}
README.md:- **Parallel computing capabilities**: PeTar leverages multi-CPU processors/threads and GPU acceleration to accelerate simulations. This enables handling of over $10^7$ particles with a $100\%$ binary fraction.
README.md:        - [Enabling GPU Acceleration](#enabling-gpu-acceleration)
README.md:    - [Using GPU](#using-gpu)
README.md:Using MPI necessitates the MPI compiler (e.g., mpic++). NVIDIA GPU and CUDA compiler are essential for GPU acceleration. SIMD support has been tested for GNU, Intel, and LLVM compilers. Since it hasn't been tested for others, it's recommended to use these three compiler types. The Fugaku ARM A64FX architecture is also compatible.
README.md:### Enabling GPU Acceleration
README.md:PeTar supports the utilization of GPUs based on the CUDA language to accelerate tree force calculations as an alternative speed-up method to SIMD acceleration. To enable this feature, use the following command:
README.md:./configure --enable-cuda
README.md:By default, the GPU is not utilized. To enable it, ensure that NVIDIA CUDA is installed and compatible with the C++ compiler.
README.md:./configure --prefix=/opt/petar --enable-cuda
README.md:This command will install the executable files in /opt/petar (this directory requires root permission) and activate GPU support.
README.md:## Using GPU
README.md:When GPU support is enabled, each MPI processor will initiate one GPU job. Modern NVIDIA GPUs can handle multiple jobs simultaneously.
README.md:Therefore, it is acceptable for `N_mpi` to exceed the number of GPUs available. However, if `N_mpi` is too large, the GPU memory may become insufficient, leading to a Cuda Memory allocation error. In such cases, utilizing more OpenMP threads and fewer MPI processors is a preferable approach.
README.md:In scenarios where multiple GPUs are present, each MPI processor will utilize a different GPU based on the processor and GPU IDs. If users wish to exclusively utilize a specific GPU, they can employ the environment variable `CUDA_VISIBLE_DEVICES=[GPU index]`. For instance:
README.md:CUDA_VISIBLE_DEVICES=1 petar [options] [particle data filename]
README.md:This command will utilize the second GPU in the system (indexing starts from 0). The `CUDA_VISIBLE_DEVICES` environment variable can also be configured in the initial script file of the terminal.
README.md:2. Enabled features (selected during configuration), such as stellar evolution packages, external packages, and GPU utilization, are listed:
README.md:| Profile  | Performance metrics of code parts   | GPU usage: `use_gpu=[True, False]`   | data.prof.rank.[MPI rank]                                   |
README.md:- **Soft Force Calculation Kernel**: SIMD (such as AVX, AVX2, AVX512, A64FX) or GPU (CUDA) technologies are employed for soft force calculations.
README.md:When appropriate tree time steps and changeover radii are set, the performance of hard calculations can be comparable to that of soft force calculations. In such cases, the use of GPUs may not significantly enhance performance, unlike in the direct N-body method for soft forces. Therefore, for optimal performance with large $N$, it is advisable to utilize more CPU cores rather than a few CPU cores with GPUs. This recommendation is particularly relevant when a substantial number of binaries are present in the simulation.
tools/analysis/profile.py:class GPUProfile(DictNpArrayMix):
tools/analysis/profile.py:    """ GPU time profile for one tree force calculation
tools/analysis/profile.py:        send: host to GPU memory sending
tools/analysis/profile.py:        receive: GPU to host memory receiving
tools/analysis/profile.py:        calc_force: GPU force calculation
tools/analysis/profile.py:        if keyword arguments "use_gpu" == True:
tools/analysis/profile.py:            gpu (GPUProfile): GPU profile for tree force calculation
tools/analysis/profile.py:            use_gpu: bool (True)
tools/analysis/profile.py:                whether cuda is used 
tools/analysis/profile.py:        use_gpu=True
tools/analysis/profile.py:        if ('use_gpu' in kwargs.keys()): use_gpu=kwargs['use_gpu']
tools/analysis/profile.py:        if (use_gpu):
tools/analysis/profile.py:            keys = [['rank',np.int64], ['time',np.float64], ['nstep',np.int64], ['n_loc',np.int64], ['comp',PeTarProfile], ['comp_bar', PeTarProfile], ['tree_soft', FDPSProfile], ['tree_nb', FDPSProfile], ['gpu',GPUProfile],['count',PeTarCount]]
Makefile.in:use_gpu_cuda=@use_cuda@
Makefile.in:CUDA_INCLUDE += -I ./bse-interface
Makefile.in:ifeq ($(use_gpu_cuda),yes)
Makefile.in:CUDAFLAGS = -D PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
Makefile.in:FDPSFLAGS += $(CUDAFLAGS)
Makefile.in:NVCC = @NVCC@ @CUDAFLAGS@ -Xcompiler="$(OPTFLAGS) $(CXXFLAGS) $(CUDAFLAGS) $(MT_FLAGS)"
Makefile.in:CXXLIBS += @CUDALIBS@
Makefile.in:OBJS += build/force_gpu_cuda.o
Makefile.in:CXXFLAGS += -D USE_GPU
Makefile.in:CXXFLAGS += -D GPU_PROFILE
Makefile.in:endif # gpu cuda
Makefile.in:CUDA_INCLUDE += -I./src @PETAR_INCLUDE@
Makefile.in:	$(CXX) $(PETAR_INCLUDE) $(OPTFLAGS) $(CXXFLAGS) $(CUDAFLAGS) $(MT_FLAGS) $^ -o $@  $(CXXLIBS)
Makefile.in:build/force_gpu_cuda.o: force_gpu_cuda.cu |build
Makefile.in:	$(NVCC) $(CUDA_INCLUDE) -c $< -o $@ 
configure:CUDALIBS
configure:CUDAFLAGS
configure:use_cuda
configure:enable_cuda
configure:with_cuda_prefix
configure:  --enable-cuda           enable CUDA (GPU) acceleration support for
configure:  --with-cuda-prefix      Prefix of your CUDA installation
configure:		       CUDAFLAGS=$CUDAFLAGS" -std=c++"$with_cpp_std
configure:# cuda
configure:# Check whether --enable-cuda was given.
configure:if test "${enable_cuda+set}" = set; then :
configure:  enableval=$enable_cuda; use_cuda=yes
configure:  use_cuda=no
configure:# Check whether --with-cuda-prefix was given.
configure:if test "${with_cuda_prefix+set}" = set; then :
configure:  withval=$with_cuda_prefix;
configure:  cuda_prefix="/usr/local/cuda"
configure:#AC_ARG_WITH([cuda-sdk-prefix],
configure:#            [AS_HELP_STRING([--with-cuda-sdk-prefix],
configure:# 	                    [Prefix of your CUDA samples (SDK) installation])],
configure:#            [cuda_sdk_prefix=$withval],
configure:# 	    [cuda_sdk_prefix="/usr/local/cuda/samples"])
configure:if test "x$use_cuda" != xno; then :
configure:       use_cuda=yes
configure:for as_dir in $cuda_prefix/bin
configure:as_fn_error $? "can't find CUDA compiler nvcc, please check whether nvcc is in environment PATH or use --with-cuda-prefix to provide the PATH of CUDA installation
configure:              cuda_prefix=`which nvcc|sed 's:/bin/nvcc::'`
configure:       PROG_NAME=$PROG_NAME".gpu"
configure:       { $as_echo "$as_me:${as_lineno-$LINENO}: checking CUDA version" >&5
configure:$as_echo_n "checking CUDA version... " >&6; }
configure:       CUDA_VERSION=`$NVCC --version|awk -F ',' 'END{print $(NF-1)}'|awk '{print $2}'`
configure:       { $as_echo "$as_me:${as_lineno-$LINENO}: result: $CUDA_VERSION" >&5
configure:$as_echo "$CUDA_VERSION" >&6; }
configure:#       AC_CHECK_FILE(["$cuda_prefix/samples/common/inc/helper_cuda.h"],
configure:#                     [CUDAFLAGS=$CUDAFLAGS" -I $cuda_prefix/samples/common/inc"],
configure:#                     [AC_CHECK_FILE(["$cuda_sdk_prefix/common/inc/helper_cuda.h"],
configure:#                                    [CUDAFLAGS=$CUDAFLAGS" -I $cuda_sdk_prefix/common/inc"],
configure:#                                    [AC_CHECK_FILE(["$cuda_sdk_prefix/Common/helper_cuda.h"],
configure:#                                                   [CUDAFLAGS=$CUDAFLAGS" -I $cuda_sdk_prefix/Common"],
configure:#                                                   [ AC_MSG_FAILURE([can't find CUDA sample inc file helper_cuda.h, please provide correct cuda SDK prefix by using --with-cuda-sdk-prefix])])])])
configure:       { $as_echo "$as_me:${as_lineno-$LINENO}: checking for main in -lcudart" >&5
configure:$as_echo_n "checking for main in -lcudart... " >&6; }
configure:if ${ac_cv_lib_cudart_main+:} false; then :
configure:LIBS="-lcudart  $LIBS"
configure:  ac_cv_lib_cudart_main=yes
configure:  ac_cv_lib_cudart_main=no
configure:{ $as_echo "$as_me:${as_lineno-$LINENO}: result: $ac_cv_lib_cudart_main" >&5
configure:$as_echo "$ac_cv_lib_cudart_main" >&6; }
configure:if test "x$ac_cv_lib_cudart_main" = xyes; then :
configure:  CUDALIBS=' -lcudart'
configure:  as_ac_File=`$as_echo "ac_cv_file_"$cuda_prefix/lib64/libcudart.so"" | $as_tr_sh`
configure:{ $as_echo "$as_me:${as_lineno-$LINENO}: checking for \"$cuda_prefix/lib64/libcudart.so\"" >&5
configure:$as_echo_n "checking for \"$cuda_prefix/lib64/libcudart.so\"... " >&6; }
configure:if test -r ""$cuda_prefix/lib64/libcudart.so""; then
configure:  CUDALIBS=" -L$cuda_prefix/lib64 -lcudart"
configure:as_fn_error $? "can't find CUDA library -lcudart, please provide correct cuda PREFIX by using --with-cuda-prefix
configure:  CUDALIBS=$CUDALIBS' -lgomp'
configure:  as_ac_File=`$as_echo "ac_cv_file_"$cuda_prefix/lib64/libgomp.so"" | $as_tr_sh`
configure:{ $as_echo "$as_me:${as_lineno-$LINENO}: checking for \"$cuda_prefix/lib64/libgomp.so\"" >&5
configure:$as_echo_n "checking for \"$cuda_prefix/lib64/libgomp.so\"... " >&6; }
configure:if test -r ""$cuda_prefix/lib64/libgomp.so""; then
configure:  CUDALIBS=$CUDALIBS" -lgomp"
configure:as_fn_error $? "can't find CUDA library -lgomp, please provide correct cuda PREFIX by using --with-cuda-prefix
configure:  use_cuda=no
configure:{ $as_echo "$as_me:${as_lineno-$LINENO}:      Using GPU:         $use_cuda" >&5
configure:$as_echo "$as_me:      Using GPU:         $use_cuda" >&6;}
configure:if test "x$use_cuda" != xno; then :
configure:  { $as_echo "$as_me:${as_lineno-$LINENO}:      CUDA compiler:     $NVCC" >&5
configure:$as_echo "$as_me:      CUDA compiler:     $NVCC" >&6;}
configure:       { $as_echo "$as_me:${as_lineno-$LINENO}:      CUDA version:      $CUDA_VERSION" >&5
configure:$as_echo "$as_me:      CUDA version:      $CUDA_VERSION" >&6;}
release.note:13. Eliminated the dependency on helper_cuda.h to support systems without CUDA SDK.
src/hard.hpp:#if  ((! defined P3T_64BIT) && (defined USE_SIMD)) || (defined USE_GPU)
src/hard.hpp:#if  ((! defined P3T_64BIT) && (defined USE_SIMD)) || (defined USE_GPU)
src/hard.hpp:#if  ((! defined P3T_64BIT) && (defined USE_SIMD)) || (defined USE_GPU)
src/force_gpu_cuda.cu:#include "cuda_pointer.h"
src/force_gpu_cuda.cu:#include "force_gpu_cuda.hpp"
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:GPUProfile gpu_profile;
src/force_gpu_cuda.cu:GPUCounter gpu_counter;
src/force_gpu_cuda.cu:	N_THREAD_GPU = 32,
src/force_gpu_cuda.cu:struct EpiGPU{
src/force_gpu_cuda.cu:struct EpjGPU{
src/force_gpu_cuda.cu:struct SpjGPU{
src/force_gpu_cuda.cu:struct ForceGPU{
src/force_gpu_cuda.cu:inline __device__ ForceGPU dev_gravity_ep_ep(
src/force_gpu_cuda.cu:    EpjGPU epjj,
src/force_gpu_cuda.cu:    ForceGPU forcei)
src/force_gpu_cuda.cu:__device__ ForceGPU force_kernel_ep_ep_1walk(
src/force_gpu_cuda.cu:    EpjGPU       *jpsh,
src/force_gpu_cuda.cu:    const EpjGPU *epj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    ForceGPU      forcei,
src/force_gpu_cuda.cu:	for(int j=j_head; j<j_tail; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[tid] = ((EpjGPU *)(epj + j)) [tid];
src/force_gpu_cuda.cu:		if(j_tail-j < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:__device__ ForceGPU force_kernel_ep_ep_2walk(
src/force_gpu_cuda.cu:    EpjGPU        jpsh[2][N_THREAD_GPU],
src/force_gpu_cuda.cu:    const EpjGPU *epj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    ForceGPU      forcei,
src/force_gpu_cuda.cu:	for(int j=0; j<nj_shorter; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[0][tid] = ((EpjGPU *)(epj + jbeg0 + j)) [tid];
src/force_gpu_cuda.cu:		jpsh[1][tid] = ((EpjGPU *)(epj + jbeg1 + j)) [tid];
src/force_gpu_cuda.cu:		if(nj_shorter-j < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:	for(int j=nj_shorter; j<nj_longer; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[0][tid] = ((EpjGPU *)(epj + jbeg_longer +  j)) [tid];
src/force_gpu_cuda.cu:		if(jrem < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:__device__ ForceGPU force_kernel_ep_ep_multiwalk(
src/force_gpu_cuda.cu:    const EpjGPU *epj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    ForceGPU      forcei,
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		EpjGPU epjj = epj[id_epj[j]];
src/force_gpu_cuda.cu:		EpjGPU epjj = epj[j];
src/force_gpu_cuda.cu:    const EpiGPU * epi,
src/force_gpu_cuda.cu:    const EpjGPU * epj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    ForceGPU     * force,
src/force_gpu_cuda.cu:	ForceGPU forcei;
src/force_gpu_cuda.cu:	int t_tail = t_head + N_THREAD_GPU - 1;
src/force_gpu_cuda.cu:	__shared__ EpjGPU jpsh[2][N_THREAD_GPU];
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    SpjGPU spjj,
src/force_gpu_cuda.cu:    SpjGPU   *jpsh,
src/force_gpu_cuda.cu:    const SpjGPU *spj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:	for(int j=j_head; j<j_tail; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[tid] = ((SpjGPU *)(spj + j)) [tid];
src/force_gpu_cuda.cu:		if(j_tail-j < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:    SpjGPU        jpsh[2][N_THREAD_GPU],
src/force_gpu_cuda.cu:    const SpjGPU *spj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:	for(int j=0; j<nj_shorter; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[0][tid] = ((SpjGPU *)(spj + jbeg0 + j)) [tid];
src/force_gpu_cuda.cu:		jpsh[1][tid] = ((SpjGPU *)(spj + jbeg1 + j)) [tid];
src/force_gpu_cuda.cu:		if(nj_shorter-j < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:	for(int j=nj_shorter; j<nj_longer; j+=N_THREAD_GPU){
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		jpsh[0][tid] = ((SpjGPU *)(spj + jbeg_longer +  j)) [tid];
src/force_gpu_cuda.cu:		if(jrem < N_THREAD_GPU){
src/force_gpu_cuda.cu:			for(int jj=0; jj<N_THREAD_GPU; jj++){
src/force_gpu_cuda.cu:    const SpjGPU *spj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:		SpjGPU spjj = spj[id_spj[j]];
src/force_gpu_cuda.cu:		SpjGPU spjj = spj[j];
src/force_gpu_cuda.cu:    const EpiGPU * epi,
src/force_gpu_cuda.cu:    const SpjGPU * spj, 
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    ForceGPU     * force,
src/force_gpu_cuda.cu:	int t_tail = t_head + N_THREAD_GPU - 1;
src/force_gpu_cuda.cu:	__shared__ SpjGPU jpsh[2][N_THREAD_GPU];
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:static cudaPointer<EpiGPU>    dev_epi;
src/force_gpu_cuda.cu:static cudaPointer<EpjGPU>    dev_epj;
src/force_gpu_cuda.cu:static cudaPointer<SpjGPU>    dev_spj;
src/force_gpu_cuda.cu:static cudaPointer<ForceGPU>  dev_force;
src/force_gpu_cuda.cu:static cudaPointer<int3>      ij_disp;
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_sends;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_sendf;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_disp;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_htod;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_calc;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_retr;
src/force_gpu_cuda.cu:static cudaEvent_t cu_event_dtoh;
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:static cudaPointer<int>      dev_id_epj;
src/force_gpu_cuda.cu:static cudaPointer<int>      dev_id_spj;
src/force_gpu_cuda.cu:PS::S32 CalcForceWithLinearCutoffCUDAMultiWalk::operator()(const PS::S32 tag,
src/force_gpu_cuda.cu:        int ngpu;
src/force_gpu_cuda.cu:        cudaGetDeviceCount(&ngpu);
src/force_gpu_cuda.cu:        int device_index=my_rank % ngpu;
src/force_gpu_cuda.cu:        cudaSetDevice(device_index);
src/force_gpu_cuda.cu:        //std::cerr<<"MPI rank "<<my_rank<<" set GPU device "<<device_index<<std::endl;
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_sends);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_sendf);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_disp);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_htod);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_calc);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_retr);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_dtoh);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        gpu_profile.copy.start();
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        gpu_profile.copy.end();
src/force_gpu_cuda.cu:        cudaEventRecord(cu_event_sends);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        cudaEventRecord(cu_event_sendf);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        gpu_profile.copy.start();
src/force_gpu_cuda.cu:        if(ni_tot_reg % N_THREAD_GPU){
src/force_gpu_cuda.cu:            ni_tot_reg /= N_THREAD_GPU;
src/force_gpu_cuda.cu:            ni_tot_reg *= N_THREAD_GPU;
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        gpu_profile.copy.end();
src/force_gpu_cuda.cu:        gpu_counter.n_walk+= n_walk;
src/force_gpu_cuda.cu:        gpu_counter.n_epi += ni_tot;
src/force_gpu_cuda.cu:        gpu_counter.n_epj += nej_tot;
src/force_gpu_cuda.cu:        gpu_counter.n_spj += nsj_tot;
src/force_gpu_cuda.cu:        gpu_counter.n_call+= 1;
src/force_gpu_cuda.cu:        cudaEventRecord(cu_event_disp);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        cudaEventRecord(cu_event_htod);
src/force_gpu_cuda.cu:        int nblocks  = ni_tot_reg / N_THREAD_GPU;
src/force_gpu_cuda.cu:        int nthreads = N_THREAD_GPU;
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:        cudaEventRecord(cu_event_calc);
src/force_gpu_cuda.cu:PS::S32 CalcForceWithLinearCutoffCUDA::operator()(const PS::S32  tag,
src/force_gpu_cuda.cu:        int ngpu;
src/force_gpu_cuda.cu:        cudaGetDeviceCount(&ngpu);
src/force_gpu_cuda.cu:        int device_index=my_rank % ngpu;
src/force_gpu_cuda.cu:        cudaSetDevice(device_index);
src/force_gpu_cuda.cu:        //std::cerr<<"MPI rank "<<my_rank<<"set GPU device "<<device_index<<std::endl;
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_disp);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_htod);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_calc);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_retr);
src/force_gpu_cuda.cu:        cudaEventCreate(&cu_event_dtoh);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    gpu_profile.copy.start();
src/force_gpu_cuda.cu:    if(ni_tot_reg % N_THREAD_GPU){
src/force_gpu_cuda.cu:        ni_tot_reg /= N_THREAD_GPU;
src/force_gpu_cuda.cu:        ni_tot_reg *= N_THREAD_GPU;
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    gpu_profile.copy.end();
src/force_gpu_cuda.cu:    gpu_counter.n_walk+= n_walk;
src/force_gpu_cuda.cu:    gpu_counter.n_epi += ni_tot;
src/force_gpu_cuda.cu:    gpu_counter.n_epj += nej_tot;
src/force_gpu_cuda.cu:    gpu_counter.n_spj += nsj_tot;
src/force_gpu_cuda.cu:    gpu_counter.n_call+= 1;
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_disp);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_htod);
src/force_gpu_cuda.cu:    int nblocks  = ni_tot_reg / N_THREAD_GPU;
src/force_gpu_cuda.cu:    int nthreads = N_THREAD_GPU;
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_calc);
src/force_gpu_cuda.cu:PS::S32 RetrieveForceCUDA(const PS::S32 tag,
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_retr);
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_dtoh);
src/force_gpu_cuda.cu:    cudaEventSynchronize(cu_event_dtoh);
src/force_gpu_cuda.cu:    cudaEventElapsedTime(&send_time, cu_event_disp, cu_event_htod);
src/force_gpu_cuda.cu:    cudaEventElapsedTime(&calc_time, cu_event_htod, cu_event_calc);
src/force_gpu_cuda.cu:    cudaEventElapsedTime(&recv_time, cu_event_retr, cu_event_dtoh);
src/force_gpu_cuda.cu:    gpu_profile.send.time += 0.001f*send_time;
src/force_gpu_cuda.cu:    gpu_profile.calc.time += 0.001f*calc_time;
src/force_gpu_cuda.cu:    gpu_profile.recv.time += 0.001f*recv_time;
src/force_gpu_cuda.cu:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.cu:    cudaEventElapsedTime(&send_time, cu_event_sends, cu_event_sendf);
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_sends);
src/force_gpu_cuda.cu:    cudaEventRecord(cu_event_sendf);
src/force_gpu_cuda.cu:    gpu_profile.send.time += 0.001f*send_time;
src/force_gpu_cuda.cu:    gpu_profile.copy.start();
src/force_gpu_cuda.cu:#ifdef GPU_PROFILE
src/force_gpu_cuda.cu:    gpu_profile.copy.end();
src/cuda_pointer.h:#include <cuda.h>
src/cuda_pointer.h:#include <cuda_runtime.h>
src/cuda_pointer.h://#include <helper_cuda.h>
src/cuda_pointer.h://#define CUDA_SAFE_CALL checkCudaErrors
src/cuda_pointer.h:#define CUDA_SAFE_CALL(val) val
src/cuda_pointer.h:struct cudaPointer{
src/cuda_pointer.h:	cudaPointer(){
src/cuda_pointer.h://        ~cudaPointer(){
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMalloc(&p, size * sizeof(T)));
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMallocHost(&p, size * sizeof(T)));
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaFree(dev_pointer));
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaFreeHost(host_pointer));
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMemcpy(dev_pointer, host_pointer, count * sizeof(T), cudaMemcpyHostToDevice));
src/cuda_pointer.h:		CUDA_SAFE_CALL(cudaMemcpy(host_pointer, dev_pointer, count * sizeof(T), cudaMemcpyDeviceToHost));
src/petar.hpp:#ifdef USE_GPU
src/petar.hpp:#include"force_gpu_cuda.hpp"
src/petar.hpp:#ifdef USE_GPU
src/petar.hpp:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/petar.hpp:        tree_soft.calcForceAllAndWriteBackMultiWalkIndex(CalcForceWithLinearCutoffCUDAMultiWalk(my_rank, eps2, rout2, G),
src/petar.hpp:                                                         RetrieveForceCUDA,
src/petar.hpp:        tree_soft.calcForceAllAndWriteBackMultiWalk(CalcForceWithLinearCutoffCUDA(my_rank, eps2, rout2, G),
src/petar.hpp:                                                    RetrieveForceCUDA,
src/petar.hpp:#elif USE_SIMD // end use_gpu
src/petar.hpp:#if defined(USE_GPU) && defined(GPU_PROFILE)
src/petar.hpp:        gpu_profile.clear();
src/petar.hpp:        gpu_counter.clear();
src/petar.hpp:#if defined(USE_GPU) && defined(GPU_PROFILE)
src/petar.hpp:            std::cout<<"**** GPU time profile (local):\n";
src/petar.hpp:            gpu_profile.dumpName(std::cout);
src/petar.hpp:            gpu_counter.dumpName(std::cout);
src/petar.hpp:            gpu_profile.dump(std::cout,dn_loop);
src/petar.hpp:            gpu_counter.dump(std::cout,dn_loop);
src/petar.hpp:#if defined(USE_GPU) && defined(GPU_PROFILE)
src/petar.hpp:            gpu_profile.dump(fprofile, dn_loop, WRITE_WIDTH);
src/petar.hpp:            gpu_counter.dump(fprofile, dn_loop, WRITE_WIDTH);
src/petar.hpp:#ifdef USE_GPU
src/petar.hpp:        fout<<"Use GPU\n";
src/petar.hpp:#ifdef GPU_PROFILE
src/petar.hpp:        fout<<"Calculate GPU profile\n";
src/petar.hpp:#if defined(USE_GPU) && defined(GPU_PROFILE)
src/petar.hpp:                gpu_profile.dumpName(fprofile, WRITE_WIDTH);
src/petar.hpp:                gpu_counter.dumpName(fprofile, WRITE_WIDTH);
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:#include "force_gpu_cuda.hpp"
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    ForceSoft force_gpu[Nepi];
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:        force_gpu[i].clear();
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    PS::F64 t_gpu=0;
src/simd_test.cxx:    std::cout<<"calc GPU\n";
src/simd_test.cxx:    t_gpu -= PS::GetWtime();
src/simd_test.cxx:    ForceSoft *force_gpu_ptr = force_gpu;
src/simd_test.cxx:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/simd_test.cxx:    CalcForceWithLinearCutoffCUDAMultiWalk f_ep_ep_gpu(0, EPISoft::eps*EPISoft::eps, EPISoft::r_out*EPISoft::r_out, ForceSoft::grav_const);
src/simd_test.cxx:    f_ep_ep_gpu(1, 1, &epi_ptr, &Nepi, &id_epj_ptr, &Nepj, &id_spj_ptr, &Nspj, epj_ptr, Nepj, spj_ptr, Nspj, true);
src/simd_test.cxx:    f_ep_ep_gpu(1, 1, &epi_ptr, &Nepi, &id_epj_ptr, &Nepj, &id_spj_ptr, &Nspj, epj_ptr, Nepj, spj_ptr, Nspj, false);
src/simd_test.cxx:    CalcForceWithLinearCutoffCUDA f_ep_ep_gpu(0, EPISoft::eps*EPISoft::eps, EPISoft::r_out*EPISoft::r_out, ForceSoft::grav_const);
src/simd_test.cxx:    f_ep_ep_gpu(1, 1, &epi_ptr, &Nepi, &epj_ptr, &Nepj, &spj_ptr, &Nspj);
src/simd_test.cxx:    RetrieveForceCUDA(1, 1, &Nepi, &force_gpu_ptr);
src/simd_test.cxx:    t_gpu += PS::GetWtime();
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    PS::F64 dfmax_gpu=0, dfpmax_gpu=0;
src/simd_test.cxx:    PS::F64 nbcount_ave_gpu=0;
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:            dfmax_gpu = std::max(dfmax_gpu, df);
src/simd_test.cxx:            if(df>DF_MAX) std::cerr<<"Force diff: i="<<i<<" nosimd["<<j<<"] "<<force[i].acc[j]+force_sp[i].acc[j]<<" gpu["<<j<<"] "<<force_gpu[i].acc[j]<<std::endl;
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:        dfpmax_gpu = std::max(dfpmax_gpu, (force_sp[i].pot+force[i].pot - force_gpu[i].pot)/force_gpu[i].pot);
src/simd_test.cxx:        if(force[i].n_ngb!=force_gpu[i].n_ngb) {
src/simd_test.cxx:            std::cerr<<"Neighbor diff: i="<<i<<" nosimd "<<force[i].n_ngb<<" gpu "<<force_gpu[i].n_ngb<<std::endl;
src/simd_test.cxx:        nbcount_ave_gpu += force_gpu[i].n_ngb;
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    std::cout<<" GPU_quad";
src/simd_test.cxx:    std::cout<<" GPU_mono";
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    std::cout<<"GPU EP+SP force diff max: "<<dfmax_gpu<<" Pot diff max: "<<dfpmax_gpu<<std::endl;
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    std::cout<<" gpu: "<<nbcount_ave_gpu;
src/simd_test.cxx:#ifdef USE_GPU
src/simd_test.cxx:    std::cout<<"Time: gpu ="<<t_gpu<<" no="<<t_ep_no+t_sp_no<<" ratio="<<(t_ep_no+t_sp_no)/t_gpu<<std::endl;
src/force_gpu_cuda.hpp:#ifdef GPU_PROFILE
src/force_gpu_cuda.hpp:extern struct GPUProfile{
src/force_gpu_cuda.hpp:    GPUProfile(): 
src/force_gpu_cuda.hpp:} gpu_profile;
src/force_gpu_cuda.hpp:extern struct GPUCounter{
src/force_gpu_cuda.hpp:    GPUCounter(): 
src/force_gpu_cuda.hpp:} gpu_counter;
src/force_gpu_cuda.hpp:#ifdef PARTICLE_SIMULATOR_GPU_MULIT_WALK_INDEX
src/force_gpu_cuda.hpp:struct CalcForceWithLinearCutoffCUDAMultiWalk{
src/force_gpu_cuda.hpp:    CalcForceWithLinearCutoffCUDAMultiWalk(){}
src/force_gpu_cuda.hpp:    CalcForceWithLinearCutoffCUDAMultiWalk(PS::S32 _rank, PS::F64 _eps2, PS::F64 _rcut2, PS::F64 _G): my_rank(_rank), eps2(_eps2), rcut2(_rcut2), G(_G) {}
src/force_gpu_cuda.hpp:struct CalcForceWithLinearCutoffCUDA{
src/force_gpu_cuda.hpp:    CalcForceWithLinearCutoffCUDA(){}
src/force_gpu_cuda.hpp:    CalcForceWithLinearCutoffCUDA(PS::S32 _rank, PS::F64 _eps2, PS::F64 _rcut2, PS::F64 _G): my_rank(_rank), eps2(_eps2), rcut2(_rcut2), G(_G) {}
src/force_gpu_cuda.hpp:PS::S32 RetrieveForceCUDA(const PS::S32 tag,

```
