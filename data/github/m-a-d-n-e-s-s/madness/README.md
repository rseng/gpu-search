# https://github.com/m-a-d-n-e-s-s/madness

```console
external/mpi.cmake:  # this is to avoid issues later consuming madness targets in codes using CUDA
README.md:* The National Science Foundation under grant NSF CHE-0625598 to the University of Tennessee, in collaboration with UIUC/NCSA. Some of the multi-threading and preliminary GPGPU ports were developed by this project.
admin/docker/ubuntu/Dockerfile:# NB to enable nvidia docker runtime *with* --runtime nvidia
admin/docker/ubuntu/Dockerfile:ENV NVIDIA_VISIBLE_DEVICES all
INSTALL.md:Linux and MacOS are supported with x86, Arm64, and IBM Power processors. GPUs are not yet utilized.
cmake/modules/EchoTargetProperty.cmake:      CUDA_PTX_COMPILATION
cmake/modules/EchoTargetProperty.cmake:      CUDA_SEPARABLE_COMPILATION
cmake/modules/EchoTargetProperty.cmake:      CUDA_RESOLVE_DEVICE_SYMBOLS
cmake/modules/EchoTargetProperty.cmake:      CUDA_RUNTIME_LIBRARY
cmake/modules/EchoTargetProperty.cmake:      CUDA_EXTENSIONS
cmake/modules/EchoTargetProperty.cmake:      CUDA_STANDARD
cmake/modules/EchoTargetProperty.cmake:      CUDA_STANDARD_REQUIRED
src/madness/external/gtest/include/gtest/internal/gtest-port.h:// with a TR1 tuple implementation.  NVIDIA's CUDA NVCC compiler
src/madness/external/gtest/include/gtest/internal/gtest-port.h:# if (defined(__GNUC__) && !defined(__CUDACC__) && (GTEST_GCC_VER_ >= 40000) \
src/madness/external/catch/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
src/madness/external/catch/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
src/madness/external/catch/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
src/madness/tensor/tensor.h:		  // Ideally use if constexpr here but want headers C++14 for cuda compatibility
src/madness/mra/graveyard://this->get_procmap().print();
src/madness/mra/graveyard:template void migrate_data<double,1>(SharedPtr<FunctionImpl<double,1,MyProcmap<1> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,1,MyProcmap<1> > > tto, DClass<1>::KeyD key);
src/madness/mra/graveyard:template void migrate_data<double,2>(SharedPtr<FunctionImpl<double,2,MyProcmap<2> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,2,MyProcmap<2> > > tto, DClass<2>::KeyD key);
src/madness/mra/graveyard:template void migrate_data<double,3>(SharedPtr<FunctionImpl<double,3,MyProcmap<3> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,3,MyProcmap<3> > > tto, DClass<3>::KeyD key);
src/madness/mra/graveyard:template void migrate_data<double,4>(SharedPtr<FunctionImpl<double,4,MyProcmap<4> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,4,MyProcmap<4> > > tto, DClass<4>::KeyD key);
src/madness/mra/graveyard:template void migrate_data<double,5>(SharedPtr<FunctionImpl<double,5,MyProcmap<5> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,5,MyProcmap<5> > > tto, DClass<5>::KeyD key);
src/madness/mra/graveyard:template void migrate_data<double,6>(SharedPtr<FunctionImpl<double,6,MyProcmap<6> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<double,6,MyProcmap<6> > > tto, DClass<6>::KeyD key);
src/madness/mra/graveyard:template void migrate<std::complex<double>,1,MyProcmap<1> >(SharedPtr<FunctionImpl<std::complex<double>,1,MyProcmap<1> > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,1,MyProcmap<1> > > tto);
src/madness/mra/graveyard:template void migrate<std::complex<double>,2,MyProcmap<2> >(SharedPtr<FunctionImpl<std::complex<double>,2,MyProcmap<2> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,2,MyProcmap<2> > > tto);
src/madness/mra/graveyard:template void migrate<std::complex<double>,3,MyProcmap<3> >(SharedPtr<FunctionImpl<std::complex<double>,3,MyProcmap<3> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,3,MyProcmap<3> > > tto);
src/madness/mra/graveyard:template void migrate<std::complex<double>,4,MyProcmap<4> >(SharedPtr<FunctionImpl<std::complex<double>,4,MyProcmap<4> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,4,MyProcmap<4> > > tto);
src/madness/mra/graveyard:template void migrate<std::complex<double>,5,MyProcmap<5> >(SharedPtr<FunctionImpl<std::complex<double>,5,MyProcmap<5> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,5,MyProcmap<5> > > tto);
src/madness/mra/graveyard:template void migrate<std::complex<double>,6,MyProcmap<6> >(SharedPtr<FunctionImpl<std::complex<double>,6,MyProcmap<6> > > tfrom, 
src/madness/mra/graveyard:	SharedPtr<FunctionImpl<std::complex<double>,6,MyProcmap<6> > > tto);
src/madness/mra/graveyard:template class LoadBalImpl<double,1,MyProcmap<1> >;
src/madness/mra/graveyard:template class LoadBalImpl<double,2,MyProcmap<2> >;
src/madness/mra/graveyard:template class LoadBalImpl<double,3,MyProcmap<3> >;
src/madness/mra/graveyard:template class LoadBalImpl<double,4,MyProcmap<4> >;
src/madness/mra/graveyard:template class LoadBalImpl<double,5,MyProcmap<5> >;
src/madness/mra/graveyard:template class LoadBalImpl<double,6,MyProcmap<6> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,1,MyProcmap<1> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,2,MyProcmap<2> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,3,MyProcmap<3> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,4,MyProcmap<4> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,5,MyProcmap<5> >;
src/madness/mra/graveyard:template class LoadBalImpl<std::complex<double>,6,MyProcmap<6> >;
src/madness/mra/graveyard:template class LBTree<1,MyProcmap<1> >;
src/madness/mra/graveyard:template class LBTree<2,MyProcmap<2> >;
src/madness/mra/graveyard:template class LBTree<3,MyProcmap<3> >;
src/madness/mra/graveyard:template class LBTree<4,MyProcmap<4> >;
src/madness/mra/graveyard:template class LBTree<5,MyProcmap<5> >;
src/madness/mra/graveyard:template class LBTree<6,MyProcmap<6> >;
src/madness/mra/graveyard:    typedef MyProcmap<D> MyProcMap;
src/madness/mra/graveyard:    typedef LBTree<D,MyProcMap> treeT;
src/madness/mra/graveyard:	    this->get_procmap().print();
src/madness/mra/graveyard:		f->get_procmap()));
src/madness/mra/lbdeux.h:            // Return the Procmap
src/madness/mra/mra.h:    /// Would be easy to modify this to also change the procmap here
src/madness/mra/mra.h:    /// if desired but presently it uses the same procmap as f.
src/madness/mra/tools/autocorr.mw:z&gt;0, int(phi(i,x)*phi(j,x-z),x=z..1));</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSRQaGlHNiJmKjYlSSJpR0YlSSJqR0YlSSJ6R0YlRiU2JEkpb3BlcmF0b3JHRiVJJmFycm93R0YlRiUtSSpwaWVjZXdpc2VHSSpwcm90ZWN0ZWRHRjA2JjE5JiIiIS1JJGludEdGJTYkKiYtSSRwaGlHRiU2JDkkSSJ4R0YlIiIiLUY6NiQ5JSwmRj1GPkYzISIiRj4vRj07RjQsJkYzRj5GPkY+MkY0RjMtRjY2JEY4L0Y9O0YzRj5GJUYlRiU=</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">phi := (i,x) -&gt; piecewise(x&lt;0,0,x&lt;1,sqrt(2*i+1) * P(i,2*x-1),x&gt;1,0);</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSRwaGlHNiJmKjYkSSJpR0YlSSJ4R0YlRiU2JEkpb3BlcmF0b3JHRiVJJmFycm93R0YlRiUtSSpwaWVjZXdpc2VHSSpwcm90ZWN0ZWRHRi82KDI5JSIiIUYzMkYyIiIiKiYtSSVzcXJ0R0YlNiMsJjkkIiIjRjVGNUY1LUkiUEdGJTYkRjssJkYyRjwhIiJGNUY1MkY1RjJGM0YlRiVGJQ==</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">phi := (i,x) -&gt; sqrt(2*i+1) * P(i,2*x-1);</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSRwaGlHNiJmKjYkSSJpR0YlSSJ4R0YlRiU2JEkpb3BlcmF0b3JHRiVJJmFycm93R0YlRiUqJi1JJXNxcnRHRiU2IywmOSQiIiMiIiJGNEY0LUkiUEdGJTYkRjIsJjklRjMhIiJGNEY0RiVGJUYl</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">with(orthopoly):</Text-field></Input></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">cplus := (i,j,p) -&gt; simplify(int(int(phi(i,x)*phi(j,x-y),x=y..1)*phi(p,y),y=0..1));</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSZjcGx1c0c2ImYqNiVJImlHRiVJImpHRiVJInBHRiVGJTYkSSlvcGVyYXRvckdGJUkmYXJyb3dHRiVGJS1JKXNpbXBsaWZ5R0YlNiMtSSRpbnRHRiU2JComLUYyNiQqJi1JJHBoaUdGJTYkOSRJInhHRiUiIiItRjk2JDklLCZGPEY9SSJ5R0YlISIiRj0vRjw7RkJGPUY9LUY5NiQ5JkZCRj0vRkI7IiIhRj1GJUYlRiU=</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">cminus := (i,j,p) -&gt; simplify(int(int(phi(i,x)*phi(j,x-y),x=0..y+1)*phi(p,y+1),y=-1..0));</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSdjbWludXNHNiJmKjYlSSJpR0YlSSJqR0YlSSJwR0YlRiU2JEkpb3BlcmF0b3JHRiVJJmFycm93R0YlRiUtSSlzaW1wbGlmeUdGJTYjLUkkaW50R0YlNiQqJi1GMjYkKiYtSSRwaGlHRiU2JDkkSSJ4R0YlIiIiLUY5NiQ5JSwmRjxGPUkieUdGJSEiIkY9L0Y8OyIiISwmRkJGPUY9Rj1GPS1GOTYkOSZGR0Y9L0ZCO0ZDRkZGJUYlRiU=</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input"/></Input></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">cminus(1,1,1);</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiMsJCokIiIkIyIiIiIiIyNGJyIiJg==</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input">Digits := 30;</Text-field></Input><Output><Text-field layout="Maple Output" style="2D Output"><Equation style="2D Output">NiM+SSdEaWdpdHNHNiIiI0k=</Equation></Text-field></Output></Group><Group><Input><Text-field layout="Normal" prompt="&gt; " style="Maple Input"># Note that 
src/madness/mra/mypmap.h:    /// Procmap implemented using Tree of TreeCoords
src/madness/mra/mypmap.h:        std::shared_ptr< ProcMapImpl<D> > tree_map; // for map_type 2
src/madness/mra/mypmap.h:        /// private method that builds the Tree underlying the procmap
src/madness/mra/mypmap.h:            tree_map = std::shared_ptr< ProcMapImpl<D> > (new ProcMapImpl<D>(v));
src/madness/world/test_world.cc:# ifdef PARSEC_HAVE_CUDA
src/madness/world/test_world.cc:#  include <cuda_runtime.h>
src/madness/world/test_world.cc:# ifdef PARSEC_HAVE_CUDA
src/madness/world/test_world.cc:extern void __cuda_hello_world(); // in hello_world.cu
src/madness/world/test_world.cc:class GPUHelloWorldTask : public TaskInterface {
src/madness/world/test_world.cc:      __cuda_hello_world();
src/madness/world/test_world.cc:void test_cuda0(World& world) {
src/madness/world/test_world.cc:  world.taskq.add(new GPUHelloWorldTask());
src/madness/world/test_world.cc:#ifdef PARSEC_HAVE_CUDA
src/madness/world/test_world.cc:        test_cuda0(world);
src/madness/world/meta.h:// TODO remove when it is possible to use CUDA/NVCC with C++17
src/madness/world/CMakeLists.txt:  if (TARGET PaRSEC::parsec AND PARSEC_HAVE_CUDA)
src/madness/world/CMakeLists.txt:    check_language(CUDA)
src/madness/world/CMakeLists.txt:    if(CMAKE_CUDA_COMPILER)
src/madness/world/CMakeLists.txt:      # cmake 3.17 decouples C++ and CUDA standards, see https://gitlab.kitware.com/cmake/cmake/issues/19123
src/madness/world/CMakeLists.txt:      # cmake 3.18 knows that CUDA 11 provides cuda_std_17
src/madness/world/CMakeLists.txt:      set(CMAKE_CUDA_STANDARD 17)
src/madness/world/CMakeLists.txt:      set(CMAKE_CUDA_EXTENSIONS OFF)
src/madness/world/CMakeLists.txt:      set(CMAKE_CUDA_STANDARD_REQUIRED ON)
src/madness/world/CMakeLists.txt:      set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
src/madness/world/CMakeLists.txt:      # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#constexpr-functions%5B/url%5D:
src/madness/world/CMakeLists.txt:      if (DEFINED CMAKE_CUDA_FLAGS)
src/madness/world/CMakeLists.txt:        set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr ${CMAKE_CUDA_FLAGS}")
src/madness/world/CMakeLists.txt:        set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr")
src/madness/world/CMakeLists.txt:      enable_language(CUDA)
src/madness/world/CMakeLists.txt:      add_library(MADtest_cuda hello_world.cu)
src/madness/world/CMakeLists.txt:      target_link_libraries(test_world PRIVATE MADtest_cuda)
src/madness/world/CMakeLists.txt:    endif(CMAKE_CUDA_COMPILER)
src/madness/world/hello_world.cu:// CUDA runtime
src/madness/world/hello_world.cu:#include <cuda_runtime.h>
src/madness/world/hello_world.cu:// helper functions and utilities to work with CUDA
src/madness/world/hello_world.cu://#include <helper_cuda.h>
src/madness/world/hello_world.cu:  printf("CUDA thread [%d, %d] says \"hello, world!\"\n",\
src/madness/world/hello_world.cu:void __cuda_hello_world() {
src/madness/world/hello_world.cu:  cudaGetDeviceCount(&device_count);
src/madness/world/hello_world.cu:    std::cout << "in __cuda_hello_world: device_count=" << device_count;
src/madness/world/hello_world.cu:  	  cudaDeviceProp deviceProp;
src/madness/world/hello_world.cu:      cudaGetDeviceProperties(&deviceProp, 0);
src/madness/world/hello_world.cu:    cudaDeviceSynchronize();

```
