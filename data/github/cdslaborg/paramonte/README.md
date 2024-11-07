# https://github.com/cdslaborg/paramonte

```console
cmake/FindMatlab4PM.cmake:**Tests using GPU resources**
cmake/FindMatlab4PM.cmake:  in case your MEX file is using the GPU and
cmake/FindMatlab4PM.cmake:  in order to be able to run unit tests on this MEX file, the GPU resources
cmake/FindMatlab4PM.cmake:  Matlab aware of the use of the GPU resources in the session, which can be
cmake/FindMatlab4PM.cmake:  performed by a command such as ``D = gpuDevice()`` at the beginning of
cmake/FindMatlab4PM.cmake:    containing the test (eg. GPU device initialization based on CMake
example/matlab/matlab/has/main.m:pm.matlab.show("pm.matlab.has.gpucoder()")
example/matlab/matlab/has/main.m:pm.matlab.show( pm.matlab.has.gpucoder() )
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>  the MATLAB GPU_Coder Toolbox.
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>  \interface{gpucoder}
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>      hasit = pm.matlab.has.gpucoder();
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>  \example{gpucoder}
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>  \output{gpucoder}
src/matlab/main/+pm/+matlab/+has/gpucoder.m:%>  \final{gpucoder}
src/matlab/main/+pm/+matlab/+has/gpucoder.m:function hasit = gpucoder()
src/matlab/main/+pm/+matlab/+has/gpucoder.m:    hasit = license('test', 'GPU_Coder');
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.F90:#define CUDA_ENABLED 1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri@routines.F90:    module procedure trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri@routines.F90:#undef CUDA_ENABLED
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_ASS_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmv_EXP_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_INVA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK5(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK4(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK3(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK2(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK1(matA, classA, operationA, matB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_ASS_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_INVA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK5(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK4(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK3(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK2(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK1(matA, classA, operationA, matB, nrow, roffA, coffA, incB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsv_EXP_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_ASS_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_ONOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTSA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trmm_EXP_CUDA_OTHA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_INVA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_ASS_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_INVA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_INVA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTOA_CGMB_ONOB_RK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_CK1
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK5(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK5
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK4(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK4
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK3(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK3
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK2(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK2
src/fortran/main/pm_matrixMulTri.F90:    PURE module subroutine trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK1(matA, classA, operationA, matB, alpha, nrow, ncol, roffA, coffA, roffB, coffB)
src/fortran/main/pm_matrixMulTri.F90:        !DEC$ ATTRIBUTES DLLEXPORT :: trsm_EXP_CUDA_OTUA_CGMB_ONOB_RK1
src/fortran/main/pm_distUnif.F90:    !>  Because the generate method has no loops or conditionals, it is also suitable for SIMD or GPU implementation.<br>
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#elif   (trmv_ENABLED || trsv_ENABLED || trmm_ENABLED || trsm_ENABLED) && ((CGMB_ENABLED && (CUDA_ENABLED || CUUA_ENABLED)) || (CGMA_ENABLED && (CUDB_ENABLED || CUUB_ENABLED)))
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     (CGMB_ENABLED && (CLDA_ENABLED || CUDA_ENABLED)) || (CGMA_ENABLED && (CLDB_ENABLED || CUDB_ENABLED))
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     (CUDA_ENABLED || CUUA_ENABLED) || (CUDB_ENABLED || CUUB_ENABLED)
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if         CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if             CUDA_ENABLED || CUUA_ENABLED
src/fortran/main/pm_matrixMulTri@routines.inc.F90:#if     CUDA_ENABLED || CUUA_ENABLED

```
