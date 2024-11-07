# https://github.com/copasi/COPASI

```console
copasi/GL/glext.h:/* reuse GL_SYNC_GPU_COMMANDS_COMPLETE */
copasi/GL/glext.h:/* Reuse tokens from ARB_gpu_shader5 */
copasi/GL/glext.h:/* Reuse tokens from ARB_gpu_shader_fp64 */
copasi/GL/glext.h:# define GL_SYNC_GPU_COMMANDS_COMPLETE     0x9117
copasi/GL/glext.h:#ifndef GL_ARB_gpu_shader5
copasi/GL/glext.h:#ifndef GL_ARB_gpu_shader_fp64
copasi/GL/glext.h:#ifndef GL_EXT_gpu_program_parameters
copasi/GL/glext.h:#ifndef GL_NV_gpu_program4
copasi/GL/glext.h:#ifndef GL_EXT_gpu_shader4
copasi/GL/glext.h:# define GL_BUFFER_GPU_ADDRESS_NV          0x8F1D
copasi/GL/glext.h:# define GL_GPU_ADDRESS_NV                 0x8F34
copasi/GL/glext.h:#ifndef GL_NV_gpu_program5
copasi/GL/glext.h:#ifndef GL_NV_gpu_shader5
copasi/GL/glext.h:/* These incomplete types let us declare types compatible with OpenCL's cl_context and cl_event */
copasi/GL/glext.h:/* ARB_gpu_shader5 (no entry points) */
copasi/GL/glext.h:/* ARB_gpu_shader_fp64 */
copasi/GL/glext.h:#ifndef GL_ARB_gpu_shader5
copasi/GL/glext.h:# define GL_ARB_gpu_shader5 1
copasi/GL/glext.h:#ifndef GL_ARB_gpu_shader_fp64
copasi/GL/glext.h:# define GL_ARB_gpu_shader_fp64 1
copasi/GL/glext.h:#ifndef GL_EXT_gpu_program_parameters
copasi/GL/glext.h:# define GL_EXT_gpu_program_parameters 1
copasi/GL/glext.h:#ifndef GL_NV_gpu_program4
copasi/GL/glext.h:# define GL_NV_gpu_program4 1
copasi/GL/glext.h:#ifndef GL_EXT_gpu_shader4
copasi/GL/glext.h:# define GL_EXT_gpu_shader4 1
copasi/GL/glext.h:#ifndef GL_NV_gpu_program5
copasi/GL/glext.h:# define GL_NV_gpu_program5 1
copasi/GL/glext.h:#ifndef GL_NV_gpu_shader5
copasi/GL/glext.h:# define GL_NV_gpu_shader5 1
CMakeModules/FindBLAS.cmake:##  Intel( older versions of mkl 32 and 64 bit), ACML,ACML_MP,ACML_GPU,Apple, NAS, Generic
CMakeModules/FindBLAS.cmake:     ((BLA_VENDOR STREQUAL "ACML_GPU") AND (NOT BLAS_ACML_GPU_LIB_DIRS))
CMakeModules/FindBLAS.cmake:    file( GLOB _ACML_GPU_ROOT "C:/AMD/acml*/GPGPUexamples" )
CMakeModules/FindBLAS.cmake:    file( GLOB _ACML_GPU_ROOT "/opt/acml*/GPGPUexamples" )
CMakeModules/FindBLAS.cmake:   list(GET _ACML_GPU_ROOT 0 _ACML_GPU_ROOT)
CMakeModules/FindBLAS.cmake: elseif( BLA_VENDOR STREQUAL "ACML_GPU" )
CMakeModules/FindBLAS.cmake:  foreach( BLAS_ACML_GPU_LIB_DIRS ${_ACML_GPU_LIB_DIRS})
CMakeModules/FindBLAS.cmake:     "" "acml;acml_mv;CALBLAS" "" ${BLAS_ACML_GPU_LIB_DIRS}
CMakeModules/FindCLAPACK.cmake:#  ACML_GPU,

```
