# https://github.com/Dylan-Aliberti/Kepler-Collisions

```console
CPP/Libraries/include/glm/detail/func_common.inl:#		if GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/func_common.inl:			// Another Cuda compiler bug https://github.com/g-truc/glm/issues/530
CPP/Libraries/include/glm/detail/func_common.inl:#			elif GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/func_common.inl:#			elif GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/func_common.inl:				// http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDA__MATH__DOUBLE_g13431dd2b40b51f9139cbb7f50c18fab.html#g13431dd2b40b51f9139cbb7f50c18fab
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA)) || \
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_LANG & GLM_LANG_CXX0X_FLAG) && (GLM_COMPILER & GLM_COMPILER_CUDA)) || \
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA)))
CPP/Libraries/include/glm/detail/setup.hpp:		(GLM_COMPILER & GLM_COMPILER_CUDA)))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:		((GLM_COMPILER & GLM_COMPILER_CUDA))))
CPP/Libraries/include/glm/detail/setup.hpp:#if GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/setup.hpp:#	define GLM_CUDA_FUNC_DEF __device__ __host__
CPP/Libraries/include/glm/detail/setup.hpp:#	define GLM_CUDA_FUNC_DECL __device__ __host__
CPP/Libraries/include/glm/detail/setup.hpp:#	define GLM_CUDA_FUNC_DEF
CPP/Libraries/include/glm/detail/setup.hpp:#	define GLM_CUDA_FUNC_DECL
CPP/Libraries/include/glm/detail/setup.hpp:#	elif GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/setup.hpp:#define GLM_FUNC_DECL GLM_CUDA_FUNC_DECL
CPP/Libraries/include/glm/detail/setup.hpp:#define GLM_FUNC_QUALIFIER GLM_CUDA_FUNC_DEF GLM_INLINE
CPP/Libraries/include/glm/detail/setup.hpp:#elif GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/setup.hpp:#	if GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/detail/setup.hpp:#		pragma message("GLM: CUDA compiler detected")
CPP/Libraries/include/glm/detail/compute_common.hpp:#if GLM_COMPILER & GLM_COMPILER_CUDA
CPP/Libraries/include/glm/simd/platform.h:// CUDA
CPP/Libraries/include/glm/simd/platform.h:#define GLM_COMPILER_CUDA			0x10000000
CPP/Libraries/include/glm/simd/platform.h:#define GLM_COMPILER_CUDA75			0x10000001
CPP/Libraries/include/glm/simd/platform.h:#define GLM_COMPILER_CUDA80			0x10000002
CPP/Libraries/include/glm/simd/platform.h:#define GLM_COMPILER_CUDA90			0x10000004
CPP/Libraries/include/glm/simd/platform.h:// CUDA
CPP/Libraries/include/glm/simd/platform.h:#elif defined(__CUDACC__)
CPP/Libraries/include/glm/simd/platform.h:#	if !defined(CUDA_VERSION) && !defined(GLM_FORCE_CUDA)
CPP/Libraries/include/glm/simd/platform.h:#		include <cuda.h>  // make sure version is defined since nvcc does not define it itself!
CPP/Libraries/include/glm/simd/platform.h:#	if CUDA_VERSION >= 8000
CPP/Libraries/include/glm/simd/platform.h:#		define GLM_COMPILER GLM_COMPILER_CUDA80
CPP/Libraries/include/glm/simd/platform.h:#	elif CUDA_VERSION >= 7500
CPP/Libraries/include/glm/simd/platform.h:#		define GLM_COMPILER GLM_COMPILER_CUDA75
CPP/Libraries/include/glm/simd/platform.h:#	elif CUDA_VERSION >= 7000
CPP/Libraries/include/glm/simd/platform.h:#		define GLM_COMPILER GLM_COMPILER_CUDA70
CPP/Libraries/include/glm/simd/platform.h:#	elif CUDA_VERSION < 7000
CPP/Libraries/include/glm/simd/platform.h:#		error "GLM requires CUDA 7.0 or higher"
CPP/Libraries/include/glm/gtx/string_cast.hpp:/// This extension is not supported with CUDA
CPP/Libraries/include/glm/gtx/string_cast.hpp:#if(GLM_COMPILER & GLM_COMPILER_CUDA)
CPP/Libraries/include/glm/gtx/string_cast.hpp:#	error "GLM_GTX_string_cast is not supported on CUDA compiler"
CPP/Libraries/include/glm/ext.hpp:#if !(GLM_COMPILER & GLM_COMPILER_CUDA)
CPP/Libraries/include/glad/glad.h:#define GL_SYNC_GPU_COMMANDS_COMPLETE 0x9117
CPP/Libraries/include/GLFW/glfw3.h: *  supports OpenGL ES via EGL, while Nvidia and Intel only support it via
CPP/Libraries/include/GLFW/glfw3.h: *  EGL, OpenGL and OpenGL ES libraries do not interface with the Nvidia binary
CPP/Libraries/include/GLFW/glfw3.h: *  zero, the GPU driver waits the specified number of screen updates before
CPP/Libraries/include/GLFW/glfw3.h: *  @remark Some GPU drivers do not honor the requested swap interval, either

```
