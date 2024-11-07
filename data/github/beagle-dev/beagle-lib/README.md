# https://github.com/beagle-dev/beagle-lib

```console
README.md:BEAGLE is a high-performance library that can perform the core calculations at the heart of most Bayesian and Maximum Likelihood phylogenetics packages. It can make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.
README.md:The paper describing the algorithms used for calculating likelihoods of sequences on trees using many core devices like graphics processing units (GPUs) is available from:  [http://tree.bio.ed.ac.uk/publications/390/](http://tree.bio.ed.ac.uk/publications/390/)
README.md:* [BEAGLE v3.1.0 for macOS with CUDA](https://github.com/beagle-dev/beagle-lib/releases/download/v3.1.0/BEAGLE.v3.1.0.pkg)
CMakeLists.txt:option(BUILD_OPENCL "Build beagle with OpenCL library" ON)
CMakeLists.txt:option(BUILD_CUDA "Build beagle with CUDA library" ON)
CMakeLists.txt:        ${PROJECT_SOURCE_DIR}/libhmsbeagle/GPU
CMakeLists.txt:  	cpack_add_component(cuda
CMakeLists.txt:  			DISPLAY_NAME "CUDA plugin"
CMakeLists.txt:  			DESCRIPTION "CUDA plugin")
CMakeLists.txt:  	cpack_add_component(opencl
CMakeLists.txt:  			DISPLAY_NAME "OpenCL plugin"
CMakeLists.txt:  			DESCRIPTION "OpenCL plugin")
project/beagle-vs-2017/installer/installer.vdproj:            "SourcePath" = "8:..\\x64\\Release\\hmsbeagle-cuda64-30.dll"
project/beagle-vs-2017/installer/installer.vdproj:            "TargetName" = "8:hmsbeagle-cuda64-30.dll"
project/beagle-vs-2017/installer/installer.vdproj:            "SourcePath" = "8:..\\x64\\Release\\hmsbeagle-opencl64-30.dll"
project/beagle-vs-2017/installer/installer.vdproj:            "TargetName" = "8:hmsbeagle-opencl64-30.dll"
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ProjectName>libhmsbeagle-opencl</ProjectName>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <RootNamespace>libhmsbeagle-opencl</RootNamespace>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";FW_OPENCL;_DEBUG;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <AdditionalDependencies>OpenCL.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-opencl64D-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <ProgramDatabaseFile>$(OutDir)libhmsbeagle-opencl.pdb</ProgramDatabaseFile>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";FW_OPENCL;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <AdditionalDependencies>OpenCL.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-opencl64-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\kernels\BeagleOpenCL_kernels.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Message Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">Creating OpenCL kernel header file</Message>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Command Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)\createOpenCLHeader.bat
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Outputs Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">BeagleOpenCL_kernels.h;%(Outputs)</Outputs>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Message Condition="'$(Configuration)|$(Platform)'=='Release|x64'">Creating OpenCL kernel header file</Message>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Command Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)\createOpenCLHeader.bat
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:      <Outputs Condition="'$(Configuration)|$(Platform)'=='Release|x64'">BeagleOpenCL_kernels.h;%(Outputs)</Outputs>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceOpenCL.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\OpenCLPlugin.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat::: windows script to create a single header with OpenCL 
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:cd ..\..\..\libhmsbeagle\GPU\kernels
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels4.cl
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_4.cl
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:set OUTFILE="BeagleOpenCL_kernels.h"
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:echo // auto-generated header file with OpenCL kernels code > %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:echo #ifndef __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:echo #define __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:..\..\..\project\beagle-vs-2017\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_SP_%%G kernels%%G.cl >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:..\..\..\project\beagle-vs-2017\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_DP_%%G kernels_dp_%%G.cl >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/createOpenCLHeader.bat:echo #endif 	// __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU\kernels">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceOpenCL.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\OpenCLPlugin.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\kernels\BeagleOpenCL_kernels.h">
project/beagle-vs-2017/libhmsbeagle-opencl/libhmsbeagle-opencl.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/createCUDAKernels.bat::: windows script to create cuda files for each state count
project/beagle-vs-2017/cuda-kernels/createCUDAKernels.bat:cd ..\..\..\libhmsbeagle\GPU\kernels
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <ProjectName>cuda-kernels</ProjectName>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <RootNamespace>cuda-kernels</RootNamespace>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.props" />
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Message>generate CUDA kernels</Message>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Command>$(ProjectDir)createCUDAKernels.bat</Command>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalIncludeDirectories>$(CUDA_PATH)\include;../../../;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";CUDA;_DEBUG;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-cuda64D-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalLibraryDirectories>$(CUDA_PATH)\lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <ProgramDatabaseFile>$(OutDir)libhmsbeagle-cuda.pdb</ProgramDatabaseFile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Command>$(ProjectDir)\createCUDAHeader.bat</Command>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Message>create combined CUDA header with ptx code</Message>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Message>generate CUDA kernels</Message>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Command>$(ProjectDir)createCUDAKernels.bat</Command>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CUDA_Build_Rule>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalIncludeDirectories>$(CUDA_PATH)\include;../../../;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";CUDA;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-cuda64-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalLibraryDirectories>$(CUDA_PATH)\lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Command>$(ProjectDir)\createCUDAHeader.bat</Command>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <Message>create combined CUDA header with ptx code</Message>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels128.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels16.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels192.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels32.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels48.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels64.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels80.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_128.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_16.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_192.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_32.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_4.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_48.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_64.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_80.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CompileOut Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)data\cuda\ptx\%(Filename).ptx</CompileOut>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.targets" />
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU\kernels">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_4.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_16.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_32.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_48.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_64.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_80.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_128.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels_dp_192.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels16.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels32.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels48.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels64.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels80.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels128.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels192.cu">
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/cuda-kernels/cuda-kernels.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat::: windows script to create a single header with compiled CUDA 
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:set OUTFILE="..\..\..\libhmsbeagle\GPU\kernels\BeagleCUDA_kernels.h"
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:echo // auto-generated header file with CUDA kernels code > %OUTFILE%
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:echo #ifndef __BeagleCUDA_kernels__ >> %OUTFILE%
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:echo #define __BeagleCUDA_kernels__ >> %OUTFILE%
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:..\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_SP_%%G ..\cuda-kernels\data\cuda\ptx\kernels%%G.ptx >> %OUTFILE%
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:..\cuda-kernels\bin2c.exe -p 0 -st -n KERNELS_STRING_DP_%%G ..\cuda-kernels\data\cuda\ptx\kernels_dp_%%G.ptx >> %OUTFILE%
project/beagle-vs-2017/cuda-kernels/createCUDAHeader.bat:echo #endif 	// __BeagleCUDA_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ProjectName>libhmsbeagle-cuda</ProjectName>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <RootNamespace>libhmsbeagle-cuda</RootNamespace>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.props" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalIncludeDirectories>$(CUDA_PATH)\include;../../../;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";CUDA;_DEBUG;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-cuda64D-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalLibraryDirectories>$(CUDA_PATH)\lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <ProgramDatabaseFile>$(OutDir)libhmsbeagle-cuda.pdb</ProgramDatabaseFile>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalIncludeDirectories>$(CUDA_PATH)\include;../../../;%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";CUDA;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalDependencies>cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-cuda64-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalLibraryDirectories>$(CUDA_PATH)\lib\x64;%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceCUDA.cpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\CUDAPlugin.cpp" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Release|x64'">None</CudaRuntime>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Release|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <CudaRuntime Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">None</CudaRuntime>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:      <AdditionalOptions Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">-DCUDA %(AdditionalOptions)</AdditionalOptions>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    </CudaCompile>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 10.0.targets" />
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU\kernels">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceCUDA.cpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\CUDAPlugin.cpp">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    <CudaCompile Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-cuda/libhmsbeagle-cuda.vcxproj.filters:    </CudaCompile>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat::: windows script to create a single header with OpenCL 
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:cd ..\..\..\libhmsbeagle\GPU\kernels
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels4.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_4.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:set OUTFILE="..\..\..\libhmsbeagle\GPU\kernels\BeagleOpenCL_kernels.h"
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:echo // auto-generated header file with OpenCL kernels code > %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:echo #ifndef __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:echo #define __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createOpenCLHeader.bat:echo #endif 	// __BeagleOpenCL_kernels__ >> %OUTFILE%
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat::: windows script to create a OpenCL kernel files
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat:cd ..\..\..\libhmsbeagle\GPU\kernels
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat:type ..\GPUImplDefs.h >> kernels4.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat:type ..\GPUImplDefs.h >> kernels%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat:type ..\GPUImplDefs.h >> kernels_dp_4.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/createCombinedOpenCLKernels.bat:type ..\GPUImplDefs.h >> kernels_dp_%%G.cl
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ProjectName>libhmsbeagle-opencl-altera</ProjectName>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <RootNamespace>libhmsbeagle-opencl-altera</RootNamespace>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";FW_OPENCL;FW_OPENCL_BINARY;FW_OPENCL_ALTERA;_DEBUG;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-opencl-altera64D-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <ProgramDatabaseFile>$(OutDir)libhmsbeagle-opencl-altera.pdb</ProgramDatabaseFile>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <PreprocessorDefinitions>WIN32;PACKAGE_VERSION="$(BeaglePackageVersion)";PLUGIN_VERSION="$(BeaglePluginVersion)";FW_OPENCL;FW_OPENCL_BINARY;FW_OPENCL_ALTERA;_CONSOLE;_EXPORTING;%(PreprocessorDefinitions)</PreprocessorDefinitions>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <OutputFile>$(OutDir)hmsbeagle-opencl-altera64-$(BeaglePluginVersion).dll</OutputFile>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\kernels\BeagleOpenCL_kernels.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Message Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">Creating OpenCL kernel header file</Message>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Command Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">$(ProjectDir)\createOpenCLHeader.bat
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Outputs Condition="'$(Configuration)|$(Platform)'=='Debug|x64'">BeagleOpenCL_kernels.h;%(Outputs)</Outputs>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Message Condition="'$(Configuration)|$(Platform)'=='Release|x64'">Creating OpenCL kernel header file</Message>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Command Condition="'$(Configuration)|$(Platform)'=='Release|x64'">$(ProjectDir)\createOpenCLHeader.bat
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:      <Outputs Condition="'$(Configuration)|$(Platform)'=='Release|x64'">BeagleOpenCL_kernels.h;%(Outputs)</Outputs>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceOpenCL.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\OpenCLAlteraPlugin.cpp" />
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <Filter Include="libhmsbeagle\GPU\kernels">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplDefs.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\KernelResource.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClInclude Include="..\..\..\libhmsbeagle\GPU\BeagleGPUImpl.hpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUImplHelper.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelLauncher.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\KernelResource.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\GPUInterfaceOpenCL.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <ClCompile Include="..\..\..\libhmsbeagle\GPU\OpenCLAlteraPlugin.cpp">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernels4.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsAll.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <CUDA_Build_Rule Include="..\..\..\libhmsbeagle\GPU\kernels\kernelsX.cu">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    </CUDA_Build_Rule>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\GPUInterface.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU</Filter>
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:    <CustomBuild Include="..\..\..\libhmsbeagle\GPU\kernels\BeagleOpenCL_kernels.h">
project/beagle-vs-2017/libhmsbeagle-opencl-altera/libhmsbeagle-opencl-altera.vcxproj.filters:      <Filter>libhmsbeagle\GPU\kernels</Filter>
project/beagle-vs-2017/libhmsbeagle.sln:Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "libhmsbeagle-opencl", "libhmsbeagle-opencl\libhmsbeagle-opencl.vcxproj", "{F1F21869-5443-4CCD-A38F-2E9A00E80762}"
project/beagle-vs-2017/libhmsbeagle.sln:Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "libhmsbeagle-opencl-altera", "libhmsbeagle-opencl-altera\libhmsbeagle-opencl-altera.vcxproj", "{8293E262-1EBB-4007-B979-B7E92D37ADB4}"
project/beagle-vs-2017/libhmsbeagle.sln:Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "libhmsbeagle-cuda", "libhmsbeagle-cuda\libhmsbeagle-cuda.vcxproj", "{94DD4F9A-6D41-44B2-A141-72CBD8452C50}"
project/beagle-vs-2017/libhmsbeagle.sln:Project("{8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942}") = "cuda-kernels", "cuda-kernels\cuda-kernels.vcxproj", "{71A686A8-26F0-44A3-A82F-4D35A33E83C1}"
project/beagle-macos/README.txt:BEAGLE is a high-performance library that can perform the core calculations at the heart of most Bayesian and Maximum Likelihood phylogenetics packages. It can make use of highly-parallel processors such as those in graphics cards (GPUs) found in many PCs.
project/beagle-macos/README.txt:The paper describing the algorithms used for calculating likelihoods of sequences on trees using many core devices like graphics processing units (GPUs) is available from:  [http://tree.bio.ed.ac.uk/publications/390/](http://tree.bio.ed.ac.uk/publications/390/)
project/beagle-macos/README.txt:* [BEAGLE v3.1.0 for macOS with CUDA](https://github.com/beagle-dev/beagle-lib/releases/download/v3.1.0/BEAGLE.v3.1.0.pkg)
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_cuda64_31.dll</td><td>{1B9512B4-2F3B-40F8-A246-1750821CDF68}</td><td>INSTALLDIR</td><td>2</td><td/><td>hmsbeagle_cuda64_31.dll</td><td>17</td><td/><td/><td/><td>/LogFile=</td><td>/LogFile=</td><td>/LogFile=</td><td>/LogFile=</td></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_opencl64_31.dll</td><td>{DF3EF185-EFAA-45AD-92FD-9E5EF3087859}</td><td>INSTALLDIR</td><td>2</td><td/><td>hmsbeagle_opencl64_31.dll</td><td>17</td><td/><td/><td/><td>/LogFile=</td><td>/LogFile=</td><td>/LogFile=</td><td>/LogFile=</td></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>AlwaysInstall</td><td>hmsbeagle_cuda64_31.dll</td></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>AlwaysInstall</td><td>hmsbeagle_opencl64_31.dll</td></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_cuda64_31.dll</td><td>hmsbeagle_cuda64_31.dll</td><td>HMSBEA~1.DLL|hmsbeagle-cuda64-31.dll</td><td>0</td><td/><td/><td/><td>1</td><td>C:\Users\Daniel\Dropbox\developer\temp\beagle-3.1.0\windows-dlls\hmsbeagle-cuda64-31.dll</td><td>1</td><td/></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_opencl64_31.dll</td><td>hmsbeagle_opencl64_31.dll</td><td>HMSBEA~1.DLL|hmsbeagle-opencl64-31.dll</td><td>0</td><td/><td/><td/><td>1</td><td>C:\Users\Daniel\Dropbox\developer\temp\beagle-3.1.0\windows-dlls\hmsbeagle-opencl64-31.dll</td><td>1</td><td/></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_cuda64_31.dll</td><td/><td/><td>_285EBD0E_21F4_465A_8741_9FD1D3ECA5B1_FILTER</td><td/><td/><td/><td/></row>
project/beagle-vs-2012-installer/beagle-installer/beagle-installer.isl:		<row><td>hmsbeagle_opencl64_31.dll</td><td/><td/><td>_BC1BE159_75A3_4ACD_A4C9_69A739603BD3_FILTER</td><td/><td/><td/><td/></row>
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6407BBAE121483F100BA8C93 /* CUDA kernels */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			buildConfigurationList = 6407BBB11214840F00BA8C93 /* Build configuration list for PBXAggregateTarget "CUDA kernels" */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productName = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C6513A96C2100BFDE68 /* OpenCL kernels */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			buildConfigurationList = 64859C6713A96C2100BFDE68 /* Build configuration list for PBXAggregateTarget "OpenCL kernels" */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = "OpenCL kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productName = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020C141214A26700C76EFB /* GPUImplHelper.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7012149DC300C76EFB /* GPUImplHelper.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020C171214A26700C76EFB /* CUDAPlugin.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7712149DFA00C76EFB /* CUDAPlugin.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020C181214A26700C76EFB /* GPUInterfaceCUDA.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7912149DFA00C76EFB /* GPUInterfaceCUDA.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C7013A96EA100BFDE68 /* GPUImplHelper.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7012149DC300C76EFB /* GPUImplHelper.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C7C13A96F7600BFDE68 /* GPUInterfaceOpenCL.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7E12149E2E00C76EFB /* GPUInterfaceOpenCL.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C7D13A96F8500BFDE68 /* OpenCLPlugin.cpp in Sources */ = {isa = PBXBuildFile; fileRef = 64020B7F12149E2E00C76EFB /* OpenCLPlugin.cpp */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C8513A9717E00BFDE68 /* OpenCL.framework in Frameworks */ = {isa = PBXBuildFile; fileRef = 64859C8413A9717E00BFDE68 /* OpenCL.framework */; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			remoteInfo = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			remoteInfo = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			remoteInfo = "OpenCL kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			remoteInfo = "hmsbeagle-opencl";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B6D12149DC300C76EFB /* BeagleGPUImpl.hpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.h; lineEnding = 0; path = BeagleGPUImpl.hpp; sourceTree = "<group>"; xcLanguageSpecificationIdentifier = xcode.lang.cpp; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B6E12149DC300C76EFB /* BeagleGPUImpl.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; path = BeagleGPUImpl.h; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B6F12149DC300C76EFB /* GPUImplDefs.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; lineEnding = 0; path = GPUImplDefs.h; sourceTree = "<group>"; xcLanguageSpecificationIdentifier = xcode.lang.objcpp; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7012149DC300C76EFB /* GPUImplHelper.cpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.cpp; path = GPUImplHelper.cpp; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7112149DC300C76EFB /* GPUImplHelper.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; path = GPUImplHelper.h; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7212149DC300C76EFB /* GPUInterface.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; path = GPUInterface.h; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7712149DFA00C76EFB /* CUDAPlugin.cpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.cpp; path = CUDAPlugin.cpp; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7812149DFA00C76EFB /* CUDAPlugin.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; path = CUDAPlugin.h; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7912149DFA00C76EFB /* GPUInterfaceCUDA.cpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.cpp; lineEnding = 0; path = GPUInterfaceCUDA.cpp; sourceTree = "<group>"; xcLanguageSpecificationIdentifier = xcode.lang.cpp; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7C12149E0D00C76EFB /* kernelsX.cu */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = text; name = kernelsX.cu; path = kernels/kernelsX.cu; sourceTree = "<group>"; xcLanguageSpecificationIdentifier = xcode.lang.opencl; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7E12149E2E00C76EFB /* GPUInterfaceOpenCL.cpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.cpp; path = GPUInterfaceOpenCL.cpp; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B7F12149E2E00C76EFB /* OpenCLPlugin.cpp */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.cpp.cpp; path = OpenCLPlugin.cpp; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020B8012149E2E00C76EFB /* OpenCLPlugin.h */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = sourcecode.c.h; path = OpenCLPlugin.h; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64020C081214A0F700C76EFB /* suppress_cuda.valgrind */ = {isa = PBXFileReference; fileEncoding = 4; lastKnownFileType = text; path = suppress_cuda.valgrind; sourceTree = "<group>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6407BB9C1214832300BA8C93 /* libhmsbeagle-cuda.dylib */ = {isa = PBXFileReference; explicitFileType = "compiled.mach-o.dylib"; includeInIndex = 0; path = "libhmsbeagle-cuda.dylib"; sourceTree = BUILT_PRODUCTS_DIR; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C7913A96EA100BFDE68 /* libhmsbeagle-opencl.dylib */ = {isa = PBXFileReference; explicitFileType = "compiled.mach-o.dylib"; includeInIndex = 0; path = "libhmsbeagle-opencl.dylib"; sourceTree = BUILT_PRODUCTS_DIR; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C8413A9717E00BFDE68 /* OpenCL.framework */ = {isa = PBXFileReference; lastKnownFileType = wrapper.framework; name = OpenCL.framework; path = /System/Library/Frameworks/OpenCL.framework; sourceTree = "<absolute>"; };
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C8513A9717E00BFDE68 /* OpenCL.framework in Frameworks */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6407BB9C1214832300BA8C93 /* libhmsbeagle-cuda.dylib */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C7913A96EA100BFDE68 /* libhmsbeagle-opencl.dylib */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6425C83C12108D5900E7ED58 /* GPU */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6425C83C12108D5900E7ED58 /* GPU */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B6F12149DC300C76EFB /* GPUImplDefs.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B6E12149DC300C76EFB /* BeagleGPUImpl.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B6D12149DC300C76EFB /* BeagleGPUImpl.hpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7112149DC300C76EFB /* GPUImplHelper.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7012149DC300C76EFB /* GPUImplHelper.cpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7212149DC300C76EFB /* GPUInterface.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6425C83D12108D6200E7ED58 /* CUDA */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6425C83E12108D8400E7ED58 /* OpenCL */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			path = GPU;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6425C83D12108D6200E7ED58 /* CUDA */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7812149DFA00C76EFB /* CUDAPlugin.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7712149DFA00C76EFB /* CUDAPlugin.cpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7912149DFA00C76EFB /* GPUInterfaceCUDA.cpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = CUDA;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6425C83E12108D8400E7ED58 /* OpenCL */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C8413A9717E00BFDE68 /* OpenCL.framework */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B8012149E2E00C76EFB /* OpenCLPlugin.h */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7F12149E2E00C76EFB /* OpenCLPlugin.cpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020B7E12149E2E00C76EFB /* GPUInterfaceOpenCL.cpp */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = OpenCL;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				646A84C4121095E500F88378 /* GPU */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		646A84C4121095E500F88378 /* GPU */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			path = GPU;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020C081214A0F700C76EFB /* suppress_cuda.valgrind */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6407BB9B1214832300BA8C93 /* hmsbeagle-cuda */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			buildConfigurationList = 6407BBAC1214838D00BA8C93 /* Build configuration list for PBXNativeTarget "hmsbeagle-cuda" */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productName = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productReference = 6407BB9C1214832300BA8C93 /* libhmsbeagle-cuda.dylib */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C6B13A96EA100BFDE68 /* hmsbeagle-opencl */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			buildConfigurationList = 64859C7613A96EA100BFDE68 /* Build configuration list for PBXNativeTarget "hmsbeagle-opencl" */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			name = "hmsbeagle-opencl";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productName = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			productReference = 64859C7913A96EA100BFDE68 /* libhmsbeagle-opencl.dylib */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6407BBAE121483F100BA8C93 /* CUDA kernels */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				6407BB9B1214832300BA8C93 /* hmsbeagle-cuda */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C6513A96C2100BFDE68 /* OpenCL kernels */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C6B13A96EA100BFDE68 /* hmsbeagle-opencl */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/kernels4.cu",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/kernelsX.cu",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/kernelsAll.cu",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.h",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			shellScript = "\ncd $BEAGLE_SOURCE_ROOT/libhmsbeagle/GPU/kernels\n\nrm -f BeagleCUDA_kernels_xcode.ptx\nrm -f BeagleCUDA_kernels_xcode.h\n\nif [ \"$CURRENT_ARCH\" == \"i386\" ]\nthen\n    GPU_ARCH=32\nelse\n    GPU_ARCH=64\nfi\n\necho \"// auto-generated header file with CUDA kernels PTX code\" > BeagleCUDA_kernels_xcode.h\n\n# \tCompile 4-state model\necho \"Making state count = 4 SP\"\n/usr/local/cuda/bin/nvcc -D_POSIX_C_SOURCE -o BeagleCUDA_kernels_xcode.ptx -ptx -DSTATE_COUNT=4 ./kernels4.cu -O3 -m$GPU_ARCH -I$BEAGLE_SOURCE_ROOT -DCUDA\necho \"#define KERNELS_STRING_SP_4 \\\"\\\\n\\\\\" >> BeagleCUDA_kernels_xcode.h\ncat BeagleCUDA_kernels_xcode.ptx | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleCUDA_kernels_xcode.h\necho \"\\\"\" >> BeagleCUDA_kernels_xcode.h\n\n#\n#\tHERE IS THE LOOP FOR GENERIC KERNELS\n#\n\nfor s in 16 32 48 64 80 128 192\ndo\necho \"Making state count = $s SP\"\n/usr/local/cuda/bin/nvcc -D_POSIX_C_SOURCE -o BeagleCUDA_kernels_xcode.ptx -ptx -DSTATE_COUNT=$s ./kernelsX.cu -O3 -m$GPU_ARCH -I$BEAGLE_SOURCE_ROOT -DCUDA\necho \"#define KERNELS_STRING_SP_$s \\\"\\\\n\\\\\" >> BeagleCUDA_kernels_xcode.h\ncat BeagleCUDA_kernels_xcode.ptx | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleCUDA_kernels_xcode.h\necho \"\\\"\" >> BeagleCUDA_kernels_xcode.h\ndone\n\n# DOUBLE PRECISION\n\n# \tCompile 4-state model\necho \"Making state count = 4 DP\"\n/usr/local/cuda/bin/nvcc -D_POSIX_C_SOURCE -o BeagleCUDA_kernels_xcode.ptx -ptx -arch compute_13 -DSTATE_COUNT=4 -DDOUBLE_PRECISION ./kernels4.cu -O3 -m$GPU_ARCH -I$BEAGLE_SOURCE_ROOT -DCUDA\necho \"#define KERNELS_STRING_DP_4 \\\"\\\\n\\\\\" >> BeagleCUDA_kernels_xcode.h\ncat BeagleCUDA_kernels_xcode.ptx | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleCUDA_kernels_xcode.h\necho \"\\\"\" >> BeagleCUDA_kernels_xcode.h\n\n#\n#\tHERE IS THE LOOP FOR GENERIC KERNELS\n#\n\nfor s in 16 32 48 64 80 128 192\ndo\necho \"Making state count = $s DP\"\n/usr/local/cuda/bin/nvcc -D_POSIX_C_SOURCE -o BeagleCUDA_kernels_xcode.ptx -ptx -arch compute_13 -DSTATE_COUNT=$s -DDOUBLE_PRECISION ./kernelsX.cu -O3 -m$GPU_ARCH -I$BEAGLE_SOURCE_ROOT -DCUDA\necho \"#define KERNELS_STRING_DP_$s \\\"\\\\n\\\\\" >> BeagleCUDA_kernels_xcode.h\ncat BeagleCUDA_kernels_xcode.ptx | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleCUDA_kernels_xcode.h\necho \"\\\"\" >> BeagleCUDA_kernels_xcode.h\ndone\n";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.cl",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				"$(BEAGLE_SOURCE_ROOT)/libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels_xcode.h",
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			shellScript = "cd $BEAGLE_SOURCE_ROOT/libhmsbeagle/GPU/kernels\n\nrm -f BeagleOpenCL_kernels_xcode.h\n\necho \"// auto-generated header file with OpenCL kernels code\" > BeagleOpenCL_kernels_xcode.h\n\n# \tCompile 4-state model\necho \"Making state count = 4 SP\"\necho \"#define KERNELS_STRING_SP_4 \\\"\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define STATE_COUNT 4\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\ncat ../GPUImplDefs.h | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsAll.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernels4.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\necho \"\\\"\" >> BeagleOpenCL_kernels_xcode.h\n\n#\n#\tHERE IS THE LOOP FOR GENERIC KERNELS\n#\n\nfor s in 16 32 48 64 80 128 192\ndo\necho \"Making state count = $s SP\"\necho \"#define KERNELS_STRING_SP_$s \\\"\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define STATE_COUNT $s\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\ncat ../GPUImplDefs.h | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsAll.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsX.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\necho \"\\\"\" >> BeagleOpenCL_kernels_xcode.h\ndone\n\n# DOUBLE PRECISION\n\n# \tCompile 4-state model\necho \"Making state count = 4 DP\"\necho \"#define KERNELS_STRING_DP_4 \\\"\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define STATE_COUNT 4\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define DOUBLE_PRECISION\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\ncat ../GPUImplDefs.h | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsAll.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernels4.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\necho \"\\\"\" >> BeagleOpenCL_kernels_xcode.h\n\n#\n#\tHERE IS THE LOOP FOR GENERIC KERNELS\n#\n\nfor s in 16 32 48 64 80 128 192\ndo\necho \"Making state count = $s DP\"\necho \"#define KERNELS_STRING_DP_$s \\\"\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define STATE_COUNT $s\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\necho \"#define DOUBLE_PRECISION\\\\n\\\\\" >> BeagleOpenCL_kernels_xcode.h\ncat ../GPUImplDefs.h | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsAll.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\ncat kernelsX.cu | sed 's/\\\"/\\\\\"/g' | sed 's/$/\\\\n\\\\/' >> BeagleOpenCL_kernels_xcode.h\necho \"\\\"\" >> BeagleOpenCL_kernels_xcode.h\ndone\n\n\n";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020C141214A26700C76EFB /* GPUImplHelper.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020C171214A26700C76EFB /* CUDAPlugin.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64020C181214A26700C76EFB /* GPUInterfaceCUDA.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C7013A96EA100BFDE68 /* GPUImplHelper.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C7D13A96F8500BFDE68 /* OpenCLPlugin.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				64859C7C13A96F7600BFDE68 /* GPUInterfaceOpenCL.cpp in Sources */,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			target = 6407BBAE121483F100BA8C93 /* CUDA kernels */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			target = 6407BB9B1214832300BA8C93 /* hmsbeagle-cuda */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			target = 64859C6513A96C2100BFDE68 /* OpenCL kernels */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:			target = 64859C6B13A96EA100BFDE68 /* hmsbeagle-opencl */;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:					CUDA,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				HEADER_SEARCH_PATHS = /usr/local/cuda/include;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				LIBRARY_SEARCH_PATHS = /usr/local/cuda/lib;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				OTHER_LDFLAGS = "-lcuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:					CUDA,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				HEADER_SEARCH_PATHS = /usr/local/cuda/include;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				LIBRARY_SEARCH_PATHS = /usr/local/cuda/lib;
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				OTHER_LDFLAGS = "-lcuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "hmsbeagle-cuda";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "CUDA kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "OpenCL kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "OpenCL kernels";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:					FW_OPENCL,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "hmsbeagle-opencl";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:					FW_OPENCL,
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:				PRODUCT_NAME = "hmsbeagle-opencl";
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6407BBAC1214838D00BA8C93 /* Build configuration list for PBXNativeTarget "hmsbeagle-cuda" */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		6407BBB11214840F00BA8C93 /* Build configuration list for PBXAggregateTarget "CUDA kernels" */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C6713A96C2100BFDE68 /* Build configuration list for PBXAggregateTarget "OpenCL kernels" */ = {
project/beagle-xcode/beagle-xcode.xcodeproj/project.pbxproj:		64859C7613A96EA100BFDE68 /* Build configuration list for PBXNativeTarget "hmsbeagle-opencl" */ = {
project/beagle-xcode/beagle-package.pmdoc/06libhmsbeagle.xml:<pkgref spec="1.12" uuid="F1303661-6034-4C62-8597-0D68051358C2"><config><identifier>hms.beagle-lib.beagleV212.libhmsbeagle-cuda.21.pkg</identifier><version>1.0</version><description></description><post-install type="none"/><requireAuthorization/><installFrom>/usr/local/lib/libhmsbeagle-cuda.21.so</installFrom><installTo mod="true">/usr/local/lib</installTo><flags><followSymbolicLinks/></flags><packageStore type="internal"></packageStore><mod>installTo</mod><mod>installTo.path</mod><mod>parent</mod></config><contents><file-list>06libhmsbeagle-contents.xml</file-list><filter>/CVS$</filter><filter>/\.svn$</filter><filter>/\.cvsignore$</filter><filter>/\.cvspass$</filter><filter>/\.DS_Store$</filter></contents></pkgref>
project/beagle-xcode/beagle-package.pmdoc/07libhmsbeagle-contents.xml:<pkg-contents spec="1.12"><f n="libhmsbeagle-opencl.21.so" o="root" g="admin" p="33261" pt="/usr/local/lib/libhmsbeagle-opencl.21.so" m="false" t="file"/></pkg-contents>
project/beagle-xcode/beagle-package.pmdoc/06libhmsbeagle-contents.xml:<pkg-contents spec="1.12"><f n="libhmsbeagle-cuda.21.so" o="root" g="admin" p="33261" pt="/usr/local/lib/libhmsbeagle-cuda.21.so" m="false" t="file"/></pkg-contents>
project/beagle-xcode/beagle-package.pmdoc/07libhmsbeagle.xml:<pkgref spec="1.12" uuid="ECB8C6EF-95BD-4768-A708-2FE9E32AD9BB"><config><identifier>hms.beagle-lib.beagleV212.libhmsbeagle-opencl.21.pkg</identifier><version>1.0</version><description></description><post-install type="none"/><requireAuthorization/><installFrom>/usr/local/lib/libhmsbeagle-opencl.21.so</installFrom><installTo mod="true">/usr/local/lib</installTo><flags><followSymbolicLinks/></flags><packageStore type="internal"></packageStore><mod>installTo</mod><mod>installTo.path</mod><mod>parent</mod></config><contents><file-list>07libhmsbeagle-contents.xml</file-list><filter>/CVS$</filter><filter>/\.svn$</filter><filter>/\.cvsignore$</filter><filter>/\.cvspass$</filter><filter>/\.DS_Store$</filter></contents></pkgref>
INSTALL:in graphics cards (GPUs) found in many PCs.
INSTALL:* Command-lines `cmake -DBUILD_CUDA=OFF ..`,  `cmake -DBUILD_OPENCL=OFF ..` and `cmake -DBUILD_JNI=OFF`
INSTALL:  turn off automatic building of the CUDA, OpenCL and Java JNI sub-libraries, respectively.
benchmarks/v3-app-note/run_benchmarks_pll_cipres.py:    rsrc_list = ['cpu', 'cpu-threaded', 'pll', 'pll-repeats', 'gpu', 'dual-gpu', 'quadruple-gpu']
benchmarks/v3-app-note/run_benchmarks_pll_cipres.py:                            elif rsrc == 'gpu':
benchmarks/v3-app-note/run_benchmarks_pll_cipres.py:                            elif rsrc == 'dual-gpu':
benchmarks/v3-app-note/run_benchmarks_pll_cipres.py:                            elif rsrc == 'quadruple-gpu':
benchmarks/v3-app-note/run_benchmarks_pll_empirical.py:    rsrc_list = ['cpu', 'cpu-threaded', 'pll', 'pll-repeats', 'gpu', 'dual-gpu']
benchmarks/v3-app-note/run_benchmarks_pll_empirical.py:                        elif rsrc == 'gpu':
benchmarks/v3-app-note/run_benchmarks_pll_empirical.py:                        elif rsrc == 'dual-gpu':
benchmarks/v3-app-note/run_benchmarks_pll_empirical.py:                        elif rsrc == 'quadruple-gpu':
java/beagle/Beagle.java: * that an instance run on certain hardware (e.g., a GPU) or have
java/beagle/Beagle.java:     * requires the THREADING_CPP flag to be set. It has no effect on GPU-based
java/beagle/BeagleFlag.java:    PROCESSOR_GPU(1 << 16, "use GPU as main processor"),
java/beagle/BeagleFlag.java:    FRAMEWORK_CUDA(1 << 22, "use CUDA implementation with GPU resources"),
java/beagle/BeagleFlag.java:    FRAMEWORK_OPENCL(1 << 23, "use OpenCL implementation with CPU or GPU resources"),
java/beagle/BeagleFactory.java://                BeagleFlag.PROCESSOR_GPU.getMask(),
suppress_cuda.valgrind:# Suppression for CUDA
suppress_cuda.valgrind:	obj:/usr/local/cuda/lib/libcuda.dylib
libhmsbeagle/beagle.h: * that an instance run on certain hardware (e.g., a GPU) or have
libhmsbeagle/beagle.h:    BEAGLE_FLAG_PROCESSOR_GPU       = 1 << 16,   /**< Use GPU as main processor */
libhmsbeagle/beagle.h:    BEAGLE_FLAG_FRAMEWORK_CUDA      = 1 << 22,   /**< Use CUDA implementation with GPU resources */
libhmsbeagle/beagle.h:    BEAGLE_FLAG_FRAMEWORK_OPENCL    = 1 << 23,   /**< Use OpenCL implementation with GPU resources */
libhmsbeagle/beagle.h: * BEAGLE_FLAG_THREADING_CPP flag to be set. It has no effect on GPU-based
libhmsbeagle/GPU/Precision.h:#ifndef GPU_PRECISION_H_
libhmsbeagle/GPU/Precision.h:#define GPU_PRECISION_H_
libhmsbeagle/GPU/Precision.h://inline void beagleCopyFromDeviceAndCastIfNecessary(GPUInterface* gpu, F* to, const F* from, F* cache,
libhmsbeagle/GPU/Precision.h://	gpu->MemcpyDeviceToHost(to, from, sizeof(F) * kPatternCount);
libhmsbeagle/GPU/Precision.h://inline void beagleCopyFromDeviceAndCastIfNecessary(GPUInterface* gpu, T* to, const F* from, F* cache,
libhmsbeagle/GPU/Precision.h://	gpu->MemcpyDeviceToHost(cache, from, sizeof(F) * kPatternCount);
libhmsbeagle/GPU/Precision.h://    gpu->MemcpyDeviceToHost(outLogLikelihoods, dIntegrationTmp, sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/Precision.h://    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/Precision.h:#endif /* GPU_PRECISION_H_ */
libhmsbeagle/GPU/CUDAPlugin.h:#ifndef __BEAGLE_CUDA_PLUGIN_H__
libhmsbeagle/GPU/CUDAPlugin.h:#define __BEAGLE_CUDA_PLUGIN_H__
libhmsbeagle/GPU/CUDAPlugin.h:namespace gpu {
libhmsbeagle/GPU/CUDAPlugin.h:class BEAGLE_DLLEXPORT CUDAPlugin : public beagle::plugin::Plugin
libhmsbeagle/GPU/CUDAPlugin.h:	CUDAPlugin();
libhmsbeagle/GPU/CUDAPlugin.h:	~CUDAPlugin();
libhmsbeagle/GPU/CUDAPlugin.h:	CUDAPlugin( const CUDAPlugin& cp );	// disallow copy by defining this private
libhmsbeagle/GPU/CUDAPlugin.h:} // namespace gpu
libhmsbeagle/GPU/CUDAPlugin.h:#endif	// __BEAGLE_CUDA_PLUGIN_H__
libhmsbeagle/GPU/GPUImplHelper.h: * @brief GPU implementation helper functions
libhmsbeagle/GPU/GPUImplHelper.h:#ifndef __GPUImplHelper__
libhmsbeagle/GPU/GPUImplHelper.h:#define __GPUImplHelper__
libhmsbeagle/GPU/GPUImplHelper.h:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/GPUImplHelper.h:#endif // __GPUImplHelper__
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ${OpenCL_INCLUDE_DIRS}
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:add_definitions(-DFW_OPENCL)
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:#include_directories(${OpenCL_INCLUDE_DIRS})
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:    set(CMD_NAME createOpenCLHeader.bat)
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:    set(CMD_NAME make_opencl_kernels.sh)
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCL_kernels.h
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        COMMENT "Building OpenCL kernels"
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:add_custom_target(OpenKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleOpenCL_kernels.h)
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:add_library(hmsbeagle-opencl SHARED
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../BeagleGPUImpl.h
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../BeagleGPUImpl.hpp
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../GPUImplDefs.h
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../GPUImplHelper.cpp
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../GPUImplHelper.h
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../GPUInterfaceOpenCL.cpp
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../OpenCLPlugin.cpp
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:        ../OpenCLPlugin.h
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:add_dependencies(hmsbeagle-opencl OpenKernels)
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:target_link_libraries(hmsbeagle-opencl ${OpenCL_LIBRARIES})
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:install(TARGETS hmsbeagle-opencl
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:    COMPONENT opencl
libhmsbeagle/GPU/CMake_OpenCL/CMakeLists.txt:SET_TARGET_PROPERTIES(hmsbeagle-opencl
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#include <cuda.h>
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#include "libhmsbeagle/GPU/GPUImplHelper.h"
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#include "libhmsbeagle/GPU/GPUInterface.h"
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#include "libhmsbeagle/GPU/KernelResource.h"
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:namespace cuda_device {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp://static int nGpuArchCoresPerSM[] = { -1, 8, 32 };
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    sSMtoCores nGpuArchCoresPerSM[] =
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    while (nGpuArchCoresPerSM[index].SM != -1)
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            return nGpuArchCoresPerSM[index].Cores;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    return nGpuArchCoresPerSM[index-1].Cores;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#define SAFE_CUDA(call) { \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                            if(error != CUDA_SUCCESS) { \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                fprintf(stderr, "CUDA error: \"%s\" (%d) from file <%s>, line %i.\n", \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                        GetCUDAErrorDescription(error), error, __FILE__, __LINE__); \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                            SAFE_CUDA(cuCtxPushCurrent(cudaContext)); \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                            SAFE_CUDA(call); \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                            SAFE_CUDA(cuCtxPopCurrent(&cudaContext)); \
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUInterface::GPUInterface() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaDevice = (CUdevice) 0;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaContext = NULL;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaModule = NULL;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaStreams = NULL;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaEvent = NULL;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUInterface::~GPUInterface() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if (cudaStreams != NULL) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            if (cudaStreams[i] != NULL && cudaStreams[i] != CU_STREAM_LEGACY)
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                SAFE_CUDA(cuStreamDestroy(cudaStreams[i]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        free(cudaStreams);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if (cudaEvent != NULL) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuEventDestroy(cudaEvent));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if (cudaContext != NULL) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuCtxPushCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuDevicePrimaryCtxRelease(cudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:int GPUInterface::Initialize() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    // Driver init; CUDA manual: "Currently, the Flags parameter must be 0."
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if (error != CUDA_SUCCESS) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetCount(&numDevices));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    CUdevice tmpCudaDevice;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, i));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuDeviceGetAttribute(&capabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuDeviceGetAttribute(&capabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:int GPUInterface::GetDeviceCount() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::InitializeKernelResource(int paddedStateCount,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLoading kernel information for CUDA!\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::SetDevice(int deviceNumber, int paddedStateCount, int categoryCount, int paddedPatternCount, int unpaddedPatternCount, int tipCount,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGet(&cudaDevice, (*resourceMap)[deviceNumber]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    CUresult error = cuDevicePrimaryCtxRetain(&cudaContext, cudaDevice);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if(error != CUDA_SUCCESS) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        fprintf(stderr, "CUDA error: \"%s\" (%d) from file <%s>, line %i.\n",
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                GetCUDAErrorDescription(error), error, __FILE__, __LINE__);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        if (error == CUDA_ERROR_INVALID_DEVICE) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            fprintf(stderr, "(The requested CUDA device is likely set to compute exclusive mode. This mode prevents multiple processes from running on the device.)");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxSetCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuModuleLoadData(&cudaModule, kernelResource->kernelCode));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    // SAFE_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cudaStreams[0] = CU_STREAM_LEGACY;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    cuEventCreate(&cudaEvent, CU_EVENT_DISABLE_TIMING);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::ResizeStreamCount(int newStreamCount) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::ResizeStreamCount\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxSynchronize());
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    if (cudaStreams != NULL) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            if (cudaStreams[i] != NULL && cudaStreams[i] != CU_STREAM_LEGACY)
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                SAFE_CUDA(cuStreamDestroy(cudaStreams[i]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        free(cudaStreams);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        cudaStreams[0] = CU_STREAM_LEGACY;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        cudaStreams = (CUstream*) malloc(sizeof(CUstream) * numStreams);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            SAFE_CUDA(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            cudaStreams[i] = stream;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::ResizeStreamCount\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::SynchronizeHost() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::SynchronizeDevice() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuEventRecord(cudaEvent, 0));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuStreamWaitEvent(0, cudaEvent, 0));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDeviceWithIndex\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        streamRecord = cudaStreams[streamRecordIndex % numStreams];
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        streamWait   = cudaStreams[streamWaitIndex % numStreams];
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuEventRecord(cudaEvent, streamRecord));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuStreamWaitEvent(streamWait, cudaEvent, 0));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDeviceWithIndex\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUFunction GPUInterface::GetFunction(const char* functionName) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUFunction cudaFunction;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuModuleGetFunction(&cudaFunction, cudaModule, functionName));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    return cudaFunction;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr* paramPtrs;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    paramPtrs = (GPUPtr*)malloc(sizeof(GPUPtr) * totalParameterCount);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:       paramPtrs[i] = (GPUPtr)(size_t)va_arg(parameters, GPUPtr);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                             cudaStreams[0], params, NULL));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernelConcurrent\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPushCurrent(cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr* paramPtrs;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    paramPtrs = (GPUPtr*)malloc(sizeof(GPUPtr) * totalParameterCount);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:       paramPtrs[i] = (GPUPtr)(size_t)va_arg(parameters, GPUPtr);
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:            SAFE_CUDA(cuStreamSynchronize(cudaStreams[waitIndexMod]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                 cudaStreams[streamIndexMod], params, NULL));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        SAFE_CUDA(cuLaunchKernel(deviceFunction, grid.x, grid.y, grid.z,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                 cudaStreams[0], params, NULL));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuCtxPopCurrent(&cudaContext));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernelConcurrent\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void* GPUInterface::MallocHost(size_t memSize) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::MallocHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MallocHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void* GPUInterface::CallocHost(size_t size, size_t length) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::CallocHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CallocHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocatePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocatePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr ptr;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + memSize));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr ptr;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + length));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr ptr;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "Allocated GPU memory %llu to %llu.\n", (unsigned long long)ptr, (unsigned long long)(ptr + length));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr subPtr = dPtr + offset;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:size_t GPUInterface::AlignMemOffset(size_t offset) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::AlignMemOffset\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AlignMemOffset\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::MemsetShort(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetShort\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetShort\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuMemcpyHtoDAsync(dest, src, memSize, cudaStreams[0]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::MemcpyDeviceToHost(void* dest,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                      const GPUPtr src,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuMemcpyDtoHAsync(dest, src, memSize, cudaStreams[0]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:                                        GPUPtr src,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUPP(cuMemcpyDtoDAsync(dest, src, memSize, cudaStreams[0]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::FreeHostMemory(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::FreePinnedHostMemory(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::FreePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::FreeMemory(GPUPtr dPtr) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceHostPointer\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    GPUPtr dPtr;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceHostPointer\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:size_t GPUInterface::GetAvailableMemory() {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#if CUDA_VERSION >= 3020
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::GetDeviceName(int deviceNumber,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    CUdevice tmpCudaDevice;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetName(deviceName, nameLength, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    CUdevice tmpCudaDevice;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::GetDeviceDescription(int deviceNumber,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    CUdevice tmpCudaDevice;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGet(&tmpCudaDevice, (*resourceMap)[deviceNumber]));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:#if CUDA_VERSION >= 3020
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceTotalMem(&totalGlobalMemory, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&clockSpeed, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    SAFE_CUDA(cuDeviceGetAttribute(&mpCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, tmpCudaDevice));
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceTypeFlag\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    long deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_GPU;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceTypeFlag\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceImplementationCode\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    BeagleDeviceImplementationCodes deviceCode = BEAGLE_CUDA_DEVICE_NVIDIA_GPU;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceImplementationCode\n");
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:const char* GPUInterface::GetCUDAErrorDescription(int errorCode) {
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:    // from cuda.h
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_SUCCESS: errorDesc = "No errors"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_VALUE: errorDesc = "Invalid value"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_OUT_OF_MEMORY: errorDesc = "Out of memory"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NOT_INITIALIZED: errorDesc = "Driver not initialized"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_DEINITIALIZED: errorDesc = "Driver deinitialized"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NO_DEVICE: errorDesc = "No CUDA-capable device available"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_DEVICE: errorDesc = "Invalid device"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_IMAGE: errorDesc = "Invalid kernel image"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_CONTEXT: errorDesc = "Invalid context"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: errorDesc = "Context already current"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_MAP_FAILED: errorDesc = "Map failed"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_UNMAP_FAILED: errorDesc = "Unmap failed"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_ARRAY_IS_MAPPED: errorDesc = "Array is mapped"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_ALREADY_MAPPED: errorDesc = "Already mapped"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NO_BINARY_FOR_GPU: errorDesc = "No binary for GPU"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_ALREADY_ACQUIRED: errorDesc = "Already acquired"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NOT_MAPPED: errorDesc = "Not mapped"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_SOURCE: errorDesc = "Invalid source"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_FILE_NOT_FOUND: errorDesc = "File not found"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_INVALID_HANDLE: errorDesc = "Invalid handle"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NOT_FOUND: errorDesc = "Not found"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_NOT_READY: errorDesc = "CUDA not ready"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_LAUNCH_FAILED: errorDesc = "Launch failed"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: errorDesc = "Launch exceeded resources"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_LAUNCH_TIMEOUT: errorDesc = "Launch exceeded timeout"; break;
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: errorDesc =
libhmsbeagle/GPU/GPUInterfaceCUDA.cpp:        case CUDA_ERROR_UNKNOWN: errorDesc = "Unknown error"; break;
libhmsbeagle/GPU/BeagleGPUImpl.h: * @file BeagleGPUImpl.h
libhmsbeagle/GPU/BeagleGPUImpl.h: * @brief GPU implementation header
libhmsbeagle/GPU/BeagleGPUImpl.h:#ifndef __BeagleGPUImpl__
libhmsbeagle/GPU/BeagleGPUImpl.h:#define __BeagleGPUImpl__
libhmsbeagle/GPU/BeagleGPUImpl.h:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/BeagleGPUImpl.h:#include "libhmsbeagle/GPU/GPUInterface.h"
libhmsbeagle/GPU/BeagleGPUImpl.h:#include "libhmsbeagle/GPU/KernelLauncher.h"
libhmsbeagle/GPU/BeagleGPUImpl.h:#define BEAGLE_GPU_GENERIC	Real
libhmsbeagle/GPU/BeagleGPUImpl.h:#define BEAGLE_GPU_TEMPLATE template <typename Real>
libhmsbeagle/GPU/BeagleGPUImpl.h:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.h:	using namespace cuda_device;
libhmsbeagle/GPU/BeagleGPUImpl.h:	using namespace opencl_device;
libhmsbeagle/GPU/BeagleGPUImpl.h:namespace gpu {
libhmsbeagle/GPU/BeagleGPUImpl.h:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.h:	namespace cuda {
libhmsbeagle/GPU/BeagleGPUImpl.h:	namespace opencl {
libhmsbeagle/GPU/BeagleGPUImpl.h:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.h:class BeagleGPUImpl : public BeagleImpl {
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUInterface* gpu;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dIntegrationTmp;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dOutFirstDeriv;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dOutSecondDeriv;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dPartialsTmp;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dFirstDerivTmp;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dSecondDerivTmp;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dSumLogLikelihood;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dSumFirstDeriv;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dSumSecondDeriv;
libhmsbeagle/GPU/BeagleGPUImpl.h:	GPUPtr dMultipleDerivatives;
libhmsbeagle/GPU/BeagleGPUImpl.h:	GPUPtr dMultipleDerivativeSum;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dPatternWeights;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dBranchLengths;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dDistanceQueue;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dPtrQueue;
libhmsbeagle/GPU/BeagleGPUImpl.h:	GPUPtr dDerivativeQueue;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dMaxScalingFactors;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dIndexMaxScalingFactors;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dAccumulatedScalingFactors;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dEigenValues;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dEvec;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dIevc;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dWeights;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dFrequencies;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dScalingFactors;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dStates;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dPartials;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dMatrices;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dCompactBuffers;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dTipPartialsBuffers;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dPartialsPtrs;
libhmsbeagle/GPU/BeagleGPUImpl.h:    // GPUPtr  dPartitionOffsets;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dPatternsNewOrder;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dTipOffsets;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dTipTypes;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dPartialsOrigin;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dStatesOrigin;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dStatesSortOrigin;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr  dPatternWeightsSort;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dStatesSort;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr dRescalingTrigger;
libhmsbeagle/GPU/BeagleGPUImpl.h:    GPUPtr* dScalingFactorsMaster;
libhmsbeagle/GPU/BeagleGPUImpl.h:    BeagleGPUImpl();
libhmsbeagle/GPU/BeagleGPUImpl.h:    virtual ~BeagleGPUImpl();
libhmsbeagle/GPU/BeagleGPUImpl.h:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.h:class BeagleGPUImplFactory : public BeagleImplFactory {
libhmsbeagle/GPU/BeagleGPUImpl.h:}	// namespace gpu
libhmsbeagle/GPU/BeagleGPUImpl.h:#include "libhmsbeagle/GPU/BeagleGPUImpl.hpp"
libhmsbeagle/GPU/BeagleGPUImpl.h:#endif // __BeagleGPUImpl__
libhmsbeagle/GPU/BeagleGPUImpl.hpp: *  BeagleGPUImpl.cpp
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "libhmsbeagle/GPU/GPUImplHelper.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "libhmsbeagle/GPU/KernelLauncher.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "libhmsbeagle/GPU/GPUInterface.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "libhmsbeagle/GPU/Precision.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#include "BeagleGPUImpl.h"
libhmsbeagle/GPU/BeagleGPUImpl.hpp:namespace gpu {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    namespace cuda {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    namespace opencl {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BeagleGPUImpl<BEAGLE_GPU_GENERIC>::BeagleGPUImpl() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu = NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dIntegrationTmp = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dOutFirstDeriv = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dOutSecondDeriv = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartialsTmp = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dFirstDerivTmp = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dSecondDerivTmp = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dSumLogLikelihood = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dSumFirstDeriv = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dSumSecondDeriv = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dMultipleDerivatives = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dMultipleDerivativeSum = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPatternWeights = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dBranchLengths = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dDistanceQueue = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPtrQueue = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dDerivativeQueue = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dMaxScalingFactors = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dIndexMaxScalingFactors = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dRescalingTrigger = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BeagleGPUImpl<BEAGLE_GPU_GENERIC>::~BeagleGPUImpl() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeHostMemory(hCategoryRates[i]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dMatrices[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dEigenValues[0]); // TODO Here is where my Mac / Intel-GPU are throwing bad-exception
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dEvec[0]);        // TODO Should be save and then release just d*Origin?
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dIevc[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dWeights[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dFrequencies[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreePinnedHostMemory(hRescalingTrigger);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                    gpu->FreeMemory(dScalingFactorsMaster[i]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeMemory(dScalingFactors[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeMemory(dPatternsNewOrder);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeMemory(dTipOffsets);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeMemory(dTipTypes);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->FreeMemory(dPatternWeightsSort);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                    gpu->FreeMemory(dStatesSortOrigin);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeHostMemory(hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // gpu->FreePinnedHostMemory(hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // gpu->FreeMemory(dPartitionOffsets);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dPartialsOrigin);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dStatesOrigin);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dIntegrationTmp);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dPartialsTmp);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dSumLogLikelihood);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dSumFirstDeriv);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dFirstDerivTmp);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dOutFirstDeriv);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dSumSecondDeriv);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dSecondDerivTmp);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dOutSecondDeriv);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dMultipleDerivatives);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dMultipleDerivativeSum);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dPatternWeights);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dBranchLengths);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dDistanceQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dPtrQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dDerivativeQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dMaxScalingFactors);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dIndexMaxScalingFactors);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dAccumulatedScalingFactors);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hPtrQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hDerivativeQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hPatternWeightsCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hDistanceQueue);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hWeightsCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hFrequenciesCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hPartialsCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hStatesCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hLogLikelihoodsCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeHostMemory(hMatrixCache);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (gpu)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        delete gpu;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::createInstance(int tipCount,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::createInstance\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu = new GPUInterface();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->Initialize();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    numDevices = gpu->GetDeviceCount();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        fprintf(stderr, "Error: No GPU devices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->CreateDevice(pluginResourceNumber);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kDeviceType = gpu->GetDeviceTypeFlag(pluginResourceNumber);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kDeviceCode = gpu->GetDeviceImplementationCode(pluginResourceNumber);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // TODO: Apple OpenCL on CPU for state count > 128
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU && kPaddedStateCount > 128) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // TODO: AMD GPU implementation for high state and category counts
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if ((kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU ||
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU) &&
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kDeviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU ||
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (kDeviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SetDevice(pluginResourceNumber, kPaddedStateCount, kCategoryCount,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kFlags |= gpu->GetDeviceTypeFlag(pluginResourceNumber);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    sizeof(GPUPtr) * ptrQueueLength;  // dPtrQueue
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        size_t availableMem = gpu->GetAvailableMemory();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        // TODO: fix memory check on CUDA and implement for OpenCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kernels = new KernelLauncher(gpu);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hWeightsCache = (Real*) gpu->CallocHost(kCategoryCount, sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hFrequenciesCache = (Real*) gpu->CallocHost(kPaddedStateCount, sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsCache = (Real*) gpu->CallocHost(kPartialsSize, sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hStatesCache = (int*) gpu->CallocHost(kPaddedPatternCount, sizeof(int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hLogLikelihoodsCache = (Real*) gpu->MallocHost(kPatternCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hMatrixCache = (Real*) gpu->CallocHost(hMatrixCacheSize, sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dEvec = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dIevc = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dEigenValues = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dWeights = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dFrequencies = (GPUPtr*) calloc(sizeof(GPUPtr),kEigenDecompCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * kMatrixCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    size_t ptrIncrement = gpu->AlignMemOffset(kMatrixSize * kCategoryCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dMatricesOrigin = gpu->AllocateMemory(kMatrixCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dMatrices[i] = gpu->CreateSubPointer(dMatricesOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * kScaleBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            ptrIncrement = gpu->AlignMemOffset(kScaleBufferSize * sizeof(signed char)); // TODO: char won't work for double-precision
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr dScalingFactorsOrigin =  gpu->AllocateMemory(ptrIncrement * kScaleBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                dScalingFactors[i] = gpu->CreateSubPointer(dScalingFactorsOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dScalingFactors = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dScalingFactorsMaster = (GPUPtr*) calloc(sizeof(GPUPtr), kScaleBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            hRescalingTrigger = (int*) gpu->AllocatePinnedHostMemory(sizeof(int), false, true);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dRescalingTrigger = gpu->GetDeviceHostPointer((void*) hRescalingTrigger);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dScalingFactors = (GPUPtr*) malloc(sizeof(GPUPtr) * (kScaleBufferCount + 1));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            ptrIncrement = gpu->AlignMemOffset(kScaleBufferSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr dScalingFactorsOrigin = gpu->AllocateMemory(ptrIncrement * (kScaleBufferCount + 1));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                dScalingFactors[i] = gpu->CreateSubPointer(dScalingFactorsOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dScalingFactors[kScaleBufferCount], zeroes,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeHostMemory(zeroes);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    ptrIncrement = gpu->AlignMemOffset(kMatrixSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dEvecOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dIevcOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dEvec[i] = gpu->CreateSubPointer(dEvecOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dIevc[i] = gpu->CreateSubPointer(dIevcOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    ptrIncrement = gpu->AlignMemOffset(kEigenValuesSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dEigenValuesOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dEigenValues[i] = gpu->CreateSubPointer(dEigenValuesOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    ptrIncrement = gpu->AlignMemOffset(kCategoryCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dWeightsOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dWeights[i] = gpu->CreateSubPointer(dWeightsOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    ptrIncrement = gpu->AlignMemOffset(kPaddedStateCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dFrequenciesOrigin = gpu->AllocateMemory(kEigenDecompCount * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dFrequencies[i] = gpu->CreateSubPointer(dFrequenciesOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dIntegrationTmp = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPatternWeights = gpu->AllocateMemory(kPatternCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dSumLogLikelihood = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartialsTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartials = (GPUPtr*) calloc(sizeof(GPUPtr), bufferCountTotal);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    ptrIncrement = gpu->AlignMemOffset(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dPartialsTmpOrigin = gpu->AllocateMemory(partialsBufferCountTotal * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartialsOrigin = gpu->CreateSubPointer(dPartialsTmpOrigin, 0, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kIndexOffsetPat = gpu->AlignMemOffset(kPartialsSize * sizeof(Real)) / sizeof(Real);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    size_t ptrIncrementStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dStatesTmpOrigin;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dStatesTmpOrigin = gpu->AllocateMemory(kCompactBufferCount * ptrIncrementStates);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dStatesOrigin = gpu->CreateSubPointer(dStatesTmpOrigin, 0, ptrIncrementStates);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dStatesOrigin = (GPUPtr) NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dStates = (GPUPtr*) calloc(sizeof(GPUPtr), kBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kIndexOffsetStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int)) / sizeof(int);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dCompactBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kCompactBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dTipPartialsBuffers = (GPUPtr*) malloc(sizeof(GPUPtr) * kTipPartialsBufferCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                dCompactBuffers[i] = gpu->CreateSubPointer(dStatesTmpOrigin, ptrIncrementStates*i, ptrIncrementStates);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                dTipPartialsBuffers[i] = gpu->CreateSubPointer(dPartialsTmpOrigin, ptrIncrement*i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dPartials[i] = gpu->CreateSubPointer(dPartialsTmpOrigin, ptrIncrement*partialsSubIndex, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dBranchLengths = gpu->AllocateMemory(kBufferCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dDistanceQueue = gpu->AllocateMemory(distanceQueueLength * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hDistanceQueue = (Real*) gpu->MallocHost(distanceQueueLength *  sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPtrQueue = gpu->AllocateMemory(sizeof(unsigned int) * ptrQueueLength);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPtrQueue = (unsigned int*) gpu->MallocHost(sizeof(unsigned int) * ptrQueueLength);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dDerivativeQueue = gpu->AllocateMemory(sizeof(unsigned int) * kBufferCount * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hDerivativeQueue = (unsigned int*) gpu->MallocHost(sizeof(unsigned int) * kBufferCount * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kSitesPerIntegrateBlock = gpu->kernelResource->patternBlockSize;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    kSitesPerBlock = gpu->kernelResource->patternBlockSize;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (kDeviceType == BEAGLE_FLAG_PROCESSOR_GPU && kPaddedStateCount == 4)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:               kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU ||
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        // gpu->MemcpyHostToDevice(dPartitionOffsets, hPartitionOffsets, transferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->ResizeStreamCount(kTipCount/2 + 1);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        // gpu->ResizeStreamCount(1);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hCategoryRates[0] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPatternWeightsCache = (Real*) gpu->MallocHost(sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dMaxScalingFactors = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dIndexMaxScalingFactors = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(unsigned int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dAccumulatedScalingFactors = gpu->AllocateMemory(sizeof(int) * kScaleBufferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving BeagleGPUImpl::createInstance\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    size_t usedMemory = availableMem - gpu->GetAvailableMemory();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:void BeagleGPUImpl<BEAGLE_GPU_GENERIC>::allocateMultiGridBuffers() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartialsPtrs = (GPUPtr) gpu->AllocatePinnedHostMemory(kOpOffsetsSize, false, false);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    dPartialsPtrs = gpu->AllocateMemory(kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsPtrs = (unsigned int*) gpu->MallocHost(kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // hPartialsPtrs = (unsigned int*) gpu->AllocatePinnedHostMemory(kOpOffsetsSize, true, false);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // dPartitionOffsets = gpu->AllocateMemory(allocationSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:char* BeagleGPUImpl<double>::getInstanceName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return (char*) "CUDA-Double";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:char* BeagleGPUImpl<float>::getInstanceName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return (char*) "CUDA-Single";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:char* BeagleGPUImpl<double>::getInstanceName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return (char*) "OpenCL-Double";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:char* BeagleGPUImpl<float>::getInstanceName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return (char*) "OpenCL-Single";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getInstanceDetails(BeagleInstanceDetails* returnInfo) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kFlags |= BEAGLE_FLAG_FRAMEWORK_CUDA;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        kFlags |= BEAGLE_FLAG_FRAMEWORK_OPENCL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCPUThreadCount(int threadCount) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setCPUThreadCount\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCPUThreadCount\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTipStates(int tipIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setTipStates\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dStates[tipIndex], hStatesCache, sizeof(int) * kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipStates\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTipPartials(int tipIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setTipPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPartials[tipIndex], hPartialsCache, sizeof(Real) * kPartialsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTipPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setRootPrePartials(const int* bufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPartials(int bufferIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPartials[bufferIndex], hPartialsCache, sizeof(Real) * kPartialsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getPartials(int bufferIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hPartialsCache, dPartials[bufferIndex], sizeof(Real) * kPartialsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setEigenDecomposition(int eigenIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr,"\tEntering BeagleGPUImpl::setEigenDecomposition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dIevc[eigenIndex], Ievc, sizeof(Real) * kMatrixSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dEvec[eigenIndex], Evec, sizeof(Real) * kMatrixSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dEigenValues[eigenIndex], Eval, sizeof(Real) * kEigenValuesSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->PrintfDeviceVector(dEigenValues[eigenIndex], kEigenValuesSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->PrintfDeviceVector(dEvec[eigenIndex], kMatrixSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->PrintfDeviceVector(dIevc[eigenIndex], kPaddedStateCount * kPaddedStateCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setEigenDecomposition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setStateFrequencies(int stateFrequenciesIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setStateFrequencies\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dFrequencies[stateFrequenciesIndex], hFrequenciesCache,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setStateFrequencies\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryWeights(int categoryWeightsIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setCategoryWeights\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dWeights[categoryWeightsIndex], tmpWeights,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCategoryWeights\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryRates(const double* inCategoryRates) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updateCategoryRates\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateCategoryRates\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setCategoryRatesWithIndex(int categoryRatesIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setCategoryRatesWithIndex\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            hCategoryRates[categoryRatesIndex] = (double*) gpu->MallocHost(sizeof(double) * kCategoryCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setCategoryRatesWithIndex\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPatternWeights(const double* inPatternWeights) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternWeights\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPatternWeights, tmpWeights,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPatternWeights\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setPatternPartitions(int partitionCount,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setPatternPartitions\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->ResizeStreamCount((kTipCount/2 + 1) * kPartitionCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (!kUsingMultiGrid && ((kPaddedPatternCount/kPartitionCount >= BEAGLE_MULTI_GRID_MAX && kDeviceCode == BEAGLE_CUDA_DEVICE_NVIDIA_GPU) || kFlags & BEAGLE_FLAG_PARALLELOPS_STREAMS) && !(kFlags & BEAGLE_FLAG_PARALLELOPS_GRID)) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        useMultiGrid = false; // use streams for larger partitions on CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeHostMemory(hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // gpu->FreeMemory(dPartitionOffsets);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setPatternPartitions\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::reorderPatternsByPartition() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::reorderPatternsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dPatternsNewOrder = gpu->AllocateMemory(newOrderSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dTipTypes = gpu->AllocateMemory(kTipCount * sizeof(int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dStatesSort = (GPUPtr*) calloc(sizeof(GPUPtr), kTipCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        size_t ptrIncrementStates = gpu->AlignMemOffset(kPaddedPatternCount * sizeof(int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dStatesSortOrigin = gpu->AllocateMemory(kCompactBufferCount * ptrIncrementStates);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dStatesSortOrigin = (GPUPtr) NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                dStatesSort[i] = gpu->CreateSubPointer(dStatesSortOrigin,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dTipOffsets  = gpu->AllocateMemory(tipOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dTipOffsets, hTipOffsets, tipOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dPatternWeightsSort = gpu->AllocateMemory(kPatternCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dTipTypes, hTipTypes, kTipCount * sizeof(int));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPatternsNewOrder, hPatternsNewOrder, newOrderSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr tmpState      = dStates[i];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr tmpPartial    = dPartials[i];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dTipOffsets, hTipOffsets, tipOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr tmpPtr = dStatesOrigin;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving BeagleGPUImpl::reorderPatternsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getTransitionMatrix(int matrixIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getTransitionMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hMatrixCache, dMatrices[matrixIndex], sizeof(Real) * kMatrixSize * kCategoryCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getTransitionMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTransitionMatrix(int matrixIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setTransitionMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setDifferentialMatrix(int matrixIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setDifferentialMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setDifferentialMatrix\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setMatrixBufferImpl(int matrixIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::setTransitionMatrices(const int* matrixIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::setTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        // Copy to GPU device
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dMatrices[matrixIndex], hMatrixCache,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::setTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::convolveTransitionMatrices(const int* firstIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\t Entering BeagleGPUImpl::convolveTransitionMatrices \n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalMatrixCount * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\t Leaving BeagleGPUImpl::convolveTransitionMatrices \n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::addTransitionMatrices(const int* firstIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::transposeTransitionMatrices(
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\t Entering BeagleGPUImpl::transposeTransitionMatrices \n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalMatrixCount * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\t Leaving BeagleGPUImpl::transposeTransitionMatrices \n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatrices(int eigenIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // Set-up and call GPU kernel
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            if (kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU && kStateCount != 4) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatricesWithModelCategories(int* eigenIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr,"\tEntering BeagleGPUImpl::updateTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                // Set-up and call GPU kernel
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                // Set-up and call GPU kernel
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * count * 2);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                // Set-up and call GPU kernel
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            if (kDeviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU && kStateCount != 4) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(dMatrices[probabilityIndices[i]], kMatrixSize * kCategoryCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatrices\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updateTransitionMatricesWithMultipleModels(const int* eigenIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updateTransitionMatricesWithMultipleModels\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * totalCount * 3);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // Set-up and call GPU kernel
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updateTransitionMatricesWithMultipleModels\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePartials(const int* operations,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePrePartials(const int *operations,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePrePartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePrePartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeDerivative(const int *postBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeDerivative\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (dOutFirstDeriv == (GPUPtr)NULL) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutFirstDeriv = gpu->AllocateMemory(kPaddedPatternCount * kBufferCount * 2 * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeDerivative\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeDerivatives(const int *postBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    if (dOutFirstDeriv == (GPUPtr)NULL) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutFirstDeriv = gpu->AllocateMemory(kPaddedPatternCount * kBufferCount * 2 * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePartialsByPartition(const int* operations,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePartialsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePartialsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::updatePrePartialsByPartition(const int* operations,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::updatePrePartialsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::updatePrePartialsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPartials(bool byPartition,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::upPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr cumulativeScalingBuffer = 0;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials1 = dPartials[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials2 = dPartials[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials3 = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr tipStates1 = dStates[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr tipStates2 = dStates[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr scalingFactors = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(partials1, kPartialsSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(partials2, kPartialsSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(scalingFactors,kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(partials3, kPartialsSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->PrintfDeviceVector(partials3, kPartialsSize, 1.0, &signal, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr scalingFactorsMulti = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            if (((gridStartOp[i+1] - gridStartOp[i]) == 1) && !byPartition && (kDeviceCode != BEAGLE_OPENCL_DEVICE_AMD_GPU)) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr tipStates1 = dStates[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr tipStates2 = dStates[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr partials1 = dPartials[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr partials2 = dPartials[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr partials3 = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr matrices1 = dMatrices[child1TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr matrices2 = dMatrices[child2TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::upPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::waitForPartials(const int* /*destinationPartials*/,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::waitForPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::waitForPartials\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BeagleGPUImpl<BEAGLE_GPU_GENERIC>::transposeTransitionMatricesOnTheFly(const int *operations,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        size_t ptrIncrement = gpu->AlignMemOffset(kMatrixSize * kCategoryCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr dMatricesOrigin = gpu->AllocateMemory((kMatrixCount + operationCount) * ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToDevice(dMatricesOrigin, dMatrices[0],
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->FreeMemory(dMatrices[0]);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dMatrices = (GPUPtr*) malloc(sizeof(GPUPtr) * (kMatrixCount + operationCount));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dMatrices[i] = gpu->CreateSubPointer(dMatricesOrigin, ptrIncrement * i, ptrIncrement);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::upPrePartials(bool byPartition,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials1 = dPartials[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials2 = dPartials[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partials3 = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr tipStates1 = dStates[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr tipStates2 = dStates[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://    GPUPtr cumulativeScalingBuffer = 0;
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr matrices1 = dMatrices[child1TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr matrices2 = dMatrices[child2TransMatIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr partials1 = dPartials[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr partials2 = dPartials[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr partials3 = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr tipStates1 = dStates[child1Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr tipStates2 = dStates[child2Index];
libhmsbeagle/GPU/BeagleGPUImpl.hpp://        GPUPtr scalingFactors = (GPUPtr)NULL;
libhmsbeagle/GPU/BeagleGPUImpl.hpp://            gpu->PrintfDeviceInt(tipStates1, kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp://            gpu->PrintfDeviceVector(partials1, kPartialsSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp://            gpu->PrintfDeviceInt(tipStates2, kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp://            gpu->PrintfDeviceVector(partials2, kPartialsSize, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->Synchronize();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::accumulateScaleFactors(const int* scalingIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::accumulateScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeof(Real)*kScaleBufferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->SynchronizeDevice();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->PrintfDeviceVector(dScalingFactors[cumulativeScalingIndex], kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::accumulateScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::accumulateScaleFactorsByPartition(const int* scalingIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::accumulateScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::accumulateScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::removeScaleFactors(const int* scalingIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::removeScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeof(Real)*kScaleBufferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->SynchronizeDevice();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::removeScaleFactorsByPartition(const int* scalingIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::removeScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::removeScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::resetScaleFactors(int cumulativeScalingIndex) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::resetScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dScalingFactors[cumulativeScalingIndex] = gpu->AllocateMemory(kScaleBufferSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dScalingFactors[cumulativeScalingIndex], zeroes,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->FreeHostMemory(zeroes);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::resetScaleFactorsByPartition(int cumulativeScalingIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::resetScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:// printf("TEST THIS FUNCTION ON CUDA & OPENCL\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    Real* zeroes = (Real*) gpu->CallocHost(ptrIncrement, partitionPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dScalingFactorPartition = gpu->CreateSubPointer(dScalingFactors[cumulativeScalingIndex],
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                                                           gpu->AlignMemOffset(ptrIncrement * startPattern),
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dScalingFactorPartition,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->FreeHostMemory(zeroes);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    Real* zeroes = (Real*) gpu->CallocHost(sizeof(Real), kPaddedPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    GPUPtr dScalingFactorPartition = dScalingFactors[cumulativeScalingIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dScalingFactorPartition,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->FreeHostMemory(zeroes);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::resetScaleFactorsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::copyScaleFactors(int destScalingIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::copyScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToDevice(dScalingFactors[destScalingIndex], dScalingFactors[srcScalingIndex], sizeof(Real)*kScaleBufferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::copyScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getScaleFactors(int srcScalingIndex,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->SynchronizeHost();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getScaleFactors\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoods(const int* bufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateRootLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr dCumulativeScalingFactor;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->PrintfDeviceVector(dPartials[rootNodeIndex], kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->PrintfDeviceVector(dIntegrationTmp, kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            const GPUPtr tmpDWeights = dWeights[categoryWeightsIndices[subsetIndex]];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            const GPUPtr tmpDFrequencies = dFrequencies[stateFrequenciesIndices[subsetIndex]];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->PrintfDeviceVector(dIntegrationTmp, kPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateRootLogLikelihoodsByPartition(
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateRootLogLikelihoodsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        // gpu->PrintfDeviceVector(dPartials[rootNodeIndex], kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:// gpu->PrintfDeviceVector(dIntegrationTmp, kPaddedPatternCount, r);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateRootLogLikelihoodsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateCrossProducts(const int *postBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateCrossProducts\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateCrossProducts\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeLogLikelihoods(const int* parentBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSumFirstDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSumSecondDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dFirstDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSecondDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutFirstDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutSecondDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partialsParent = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr partialsChild = dPartials[childIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr statesChild = dStates[childIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr transMatrix = dMatrices[probIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        GPUPtr dCumulativeScalingFactor;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr firstDerivMatrix = dMatrices[firstDerivIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr secondDerivMatrix = dMatrices[firstDerivIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                // TODO: test GPU derivative matrices for statesPartials (including extra ambiguity column)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            // TODO: improve performance of GPU implementation of derivatives for calculateEdgeLnL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr firstDerivMatrix = dMatrices[firstDerivIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            GPUPtr secondDerivMatrix = dMatrices[secondDerivIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                // TODO: test GPU derivative matrices for statesPartials (including extra ambiguity column)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumSecondDeriv, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                fprintf(stderr,"BeagleGPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and SCALING_ALWAYS\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                gpu->MemcpyHostToDevice(dPtrQueue, hPtrQueue, sizeof(unsigned int) * count);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr partialsParent = dPartials[parIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr partialsChild = dPartials[childIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr statesChild = dStates[childIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                GPUPtr transMatrix = dMatrices[probIndex];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                const GPUPtr tmpDWeights = dWeights[categoryWeightsIndices[subsetIndex]];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                const GPUPtr tmpDFrequencies = dFrequencies[stateFrequenciesIndices[subsetIndex]];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:                    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            fprintf(stderr,"BeagleGPUImpl::calculateEdgeLogLikelihoods not yet implemented for count > 1 and derivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calculateEdgeLogLikelihoodsByPartition(
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::calculateEdgeLogLikelihoodsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSumFirstDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSumSecondDeriv = gpu->AllocateMemory(kSumSitesBlockCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dFirstDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dSecondDerivTmp = gpu->AllocateMemory(kPartialsSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutFirstDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dOutSecondDeriv = gpu->AllocateMemory((kPaddedPatternCount + kResultPaddedPatterns) * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->UnmapMemory(dPartialsPtrs, hPartialsPtrs);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dPartialsPtrs, hPartialsPtrs, transferSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    #ifdef FW_OPENCL
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    hPartialsPtrs = (unsigned int*)gpu->MapMemory(dPartialsPtrs, kOpOffsetsSize);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::calculateEdgeLogLikelihoodsByPartition\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getLogLikelihood(double* outSumLogLikelihood) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getLogLikelihood\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumLogLikelihood, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getLogLikelihood\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getDerivatives(double* outSumFirstDerivative,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumFirstDeriv, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dSumSecondDeriv, sizeof(Real) * kSumSitesBlockCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getSiteLogLikelihoods(double* outLogLikelihoods) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:// TODO: copy directly to outLogLikelihoods when GPU is running in double precision
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dIntegrationTmp, sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteLogLikelihoods\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::getSiteDerivatives(double* outFirstDerivatives,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tEntering BeagleGPUImpl::getSiteDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutFirstDeriv, sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hLogLikelihoodsCache, dOutSecondDeriv, sizeof(Real) * kPatternCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    fprintf(stderr, "\tLeaving  BeagleGPUImpl::getSiteDerivatives\n");
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calcEdgeFirstDerivatives(const int *postBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dDerivativeQueue, hDerivativeQueue, sizeof(unsigned int) * 3 * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivatives, sizeof(Real) * kPaddedPatternCount * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivativeSum, sizeof(Real) * length);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:void BeagleGPUImpl<BEAGLE_GPU_GENERIC>::initDerivatives(int replicates) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        if (dMultipleDerivatives != (GPUPtr)NULL) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            gpu->FreeMemory(dMultipleDerivatives);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        dMultipleDerivatives = gpu->AllocateMemory(minSize * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:        if (dMultipleDerivativeSum == (GPUPtr)NULL) {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:            dMultipleDerivativeSum = gpu->AllocateMemory(kBufferCount * sizeof(Real));
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:int BeagleGPUImpl<BEAGLE_GPU_GENERIC>::calcCrossProducts(const int *postBufferIndices,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dDerivativeQueue, hDerivativeQueue, sizeof(unsigned int) * 2 * totalCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    const GPUPtr categoryWeights = dWeights[0];
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyHostToDevice(dDistanceQueue, hDistanceQueue, sizeof(Real) * lengthCount);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    gpu->MemcpyDeviceToHost(hTmp.data(), dMultipleDerivatives, sizeof(Real) * kPaddedStateCount * kPaddedStateCount * replicates);
libhmsbeagle/GPU/BeagleGPUImpl.hpp:// BeagleGPUImplFactory public methods
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BeagleImpl*  BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::createImpl(int tipCount,
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    BeagleImpl* impl = new BeagleGPUImpl<BEAGLE_GPU_GENERIC>();
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:const char* BeagleGPUImplFactory<double>::getName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return "GPU-DP-CUDA";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:const char* BeagleGPUImplFactory<float>::getName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return "GPU-SP-CUDA";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:const char* BeagleGPUImplFactory<double>::getName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return "DP-OpenCL";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:const char* BeagleGPUImplFactory<float>::getName() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    return "SP-OpenCL";
libhmsbeagle/GPU/BeagleGPUImpl.hpp:BEAGLE_GPU_TEMPLATE
libhmsbeagle/GPU/BeagleGPUImpl.hpp:const long BeagleGPUImplFactory<BEAGLE_GPU_GENERIC>::getFlags() {
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#ifdef CUDA
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    flags |= BEAGLE_FLAG_FRAMEWORK_CUDA |
libhmsbeagle/GPU/BeagleGPUImpl.hpp:             BEAGLE_FLAG_PROCESSOR_GPU;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/BeagleGPUImpl.hpp:    flags |= BEAGLE_FLAG_FRAMEWORK_OPENCL |
libhmsbeagle/GPU/BeagleGPUImpl.hpp:             BEAGLE_FLAG_PROCESSOR_CPU | BEAGLE_FLAG_PROCESSOR_GPU | BEAGLE_FLAG_PROCESSOR_OTHER;
libhmsbeagle/GPU/BeagleGPUImpl.hpp:} // end of gpu namespace
libhmsbeagle/GPU/KernelLauncher.h: * @brief GPU kernel launcher
libhmsbeagle/GPU/KernelLauncher.h:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/KernelLauncher.h:#include "libhmsbeagle/GPU/GPUInterface.h"
libhmsbeagle/GPU/KernelLauncher.h:#ifdef CUDA
libhmsbeagle/GPU/KernelLauncher.h:	namespace cuda_device {
libhmsbeagle/GPU/KernelLauncher.h:	namespace opencl_device {
libhmsbeagle/GPU/KernelLauncher.h:    GPUInterface* gpu;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixConvolution;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixTranspose;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixMulADBMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixMulADB;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixMulADBFirstDeriv;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fMatrixMulADBSecondDeriv;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockCoherentMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockCoherentPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockCoherent;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockFixedScalingMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockFixedScalingPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockFixedScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockAutoScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockCheckScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsByPatternBlockFixedCheckScaling;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsPartialsEdgeFirstDerivatives;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsStatesEdgeFirstDerivatives;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsPartialsCrossProducts;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsStatesCrossProducts;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fMultipleNodeSiteReduction;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fMultipleNodeSiteSquaredReduction;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsPartialsGrowing;
libhmsbeagle/GPU/KernelLauncher.h:	GPUFunction fPartialsStatesGrowing;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockCoherentMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockCoherentPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockCoherent;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockFixedScalingMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockFixedScalingPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsByPatternBlockFixedScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockCoherentMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockCoherentPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockCoherent;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockFixedScalingMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockFixedScalingPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesStatesByPatternBlockFixedScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsEdgeLikelihoods;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsEdgeLikelihoodsByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsPartialsEdgeLikelihoodsSecondDeriv;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsEdgeLikelihoods;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsEdgeLikelihoodsByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fStatesPartialsEdgeLikelihoodsSecondDeriv;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsDynamicScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsDynamicScalingPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsDynamicScalingSecondDeriv;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fAccumulateFactorsDynamicScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fAccumulateFactorsDynamicScalingByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fAccumulateFactorsAutoScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fRemoveFactorsDynamicScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fRemoveFactorsDynamicScalingByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fResetFactorsDynamicScalingByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingAccumulate;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingAccumulateByPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingAccumulateDifference;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingAccumulateReciprocal;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fPartialsDynamicScalingSlow;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoods;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsPartition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsSecondDeriv;
libhmsbeagle/GPU/KernelLauncher.h:	  GPUFunction fIntegrateLikelihoodsMulti;
libhmsbeagle/GPU/KernelLauncher.h:	  GPUFunction fIntegrateLikelihoodsFixedScaleMulti;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fIntegrateLikelihoodsAutoScaling;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fSumSites1;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fSumSites1Partition;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fSumSites2;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fSumSites3;
libhmsbeagle/GPU/KernelLauncher.h:    GPUFunction fReorderPatterns;
libhmsbeagle/GPU/KernelLauncher.h:    KernelLauncher(GPUInterface* inGpu);
libhmsbeagle/GPU/KernelLauncher.h:    void ReorderPatterns(GPUPtr dPartials,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dStates,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dStatesSort,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dTipOffsets,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dTipTypes,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dPatternsNewOrder,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr dPatternWeightsSort,
libhmsbeagle/GPU/KernelLauncher.h:    void ConvolveTransitionMatrices(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:                          GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void TransposeTransitionMatrices(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:    								 GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void GetTransitionProbabilitiesSquareMulti(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void GetTransitionProbabilitiesSquare(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void GetTransitionProbabilitiesSquareFirstDeriv(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void GetTransitionProbabilitiesSquareSecondDeriv(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.h:                                          GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsPruningDynamicCheckScaling(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr* dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr* dScalingFactorsMaster,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dRescalingTrigger,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsPruningMulti(GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.h:                                      GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.h:                                      GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                      GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                               GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesPartialsPruningMulti(GPUPtr states,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesPartialsPruningDynamicScaling(GPUPtr states1,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesStatesPruningMulti(GPUPtr states,
libhmsbeagle/GPU/KernelLauncher.h:                                  GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.h:                                  GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.h:                                  GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                  GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesStatesPruningDynamicScaling(GPUPtr states1,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr states2,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                           GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.h:	void PartialsStatesGrowing(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.h:                               GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.h:                               GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                               GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                               GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsGrowing(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.h:                                 GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.h:                                 GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                 GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.h:                                 GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.h:	void PartialsStatesEdgeFirstDerivatives(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr states0,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr partials0,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr matrices0,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.h:	void PartialsPartialsEdgeFirstDerivatives(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr partials0,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr matrices0,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.h:											  GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.h:	void PartialsStatesCrossProducts(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.h:									 GPUPtr states0,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr lengths,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr categoryWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr patternWeights,
libhmsbeagle/GPU/KernelLauncher.h:	void PartialsPartialsCrossProducts(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr lengths,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr categoryWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                                   GPUPtr patternWeights,
libhmsbeagle/GPU/KernelLauncher.h:	void MultipleNodeSiteReduction(GPUPtr outSiteValues,
libhmsbeagle/GPU/KernelLauncher.h:								   GPUPtr inSiteValues,
libhmsbeagle/GPU/KernelLauncher.h:								   GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.h:	void MultipleNodeSiteSquaredReduction(GPUPtr outSiteValues,
libhmsbeagle/GPU/KernelLauncher.h:								          GPUPtr inSiteValues,
libhmsbeagle/GPU/KernelLauncher.h:								          GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsDynamicScalingPartition(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                                     GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsAutoScaling(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                            GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsDynamicScalingSecondDeriv(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dFirstDerivResult,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dSecondDerivResult,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dRootFirstDeriv,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dRootSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                                       GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dChildParials,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dPartialsOrigin,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dMatricesOrigin,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void PartialsPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dFirstDerivTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dSecondDerivTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dChildParials,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dFirstDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dSecondDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dChildStates,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dPartialsOrigin,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dStatesOrigin,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dMatricesOrigin,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void StatesPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dFirstDerivTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dSecondDerivTmp,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dChildStates,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dFirstDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.h:                                                  GPUPtr dSecondDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.h:    void AccumulateFactorsDynamicScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void AccumulateFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                                    GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void AccumulateFactorsAutoScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                      GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                      GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void RemoveFactorsDynamicScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                     GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                     GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void RemoveFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                                GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:                                                GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void ResetFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void RescalePartials(GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                         GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.h:    void RescalePartialsByPartition(GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:                                    GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoods(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                              GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                              GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                              GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsPartition(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                       GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.h:    void IntegrateLikelihoodsSecondDeriv(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dFirstDerivResult,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dSecondDerivResult,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dRootFirstDeriv,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dRootSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:                                         GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:	void IntegrateLikelihoodsMulti(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:								   GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:								   GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:								   GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:	void IntegrateLikelihoodsFixedScaleMulti(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.h:                                             GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dMaxScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:											 GPUPtr dIndexMaxScalingFactors,
libhmsbeagle/GPU/KernelLauncher.h:    void SumSites1(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.h:    void SumSites1Partition(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.h:                            GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.h:                            GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.h:    void SumSites2(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dArray2,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum2,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.h:    void SumSites3(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dArray2,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum2,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dArray3,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dSum3,
libhmsbeagle/GPU/KernelLauncher.h:                  GPUPtr dPatternWeights,
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:#include "libhmsbeagle/GPU/OpenCLAlteraPlugin.h"
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:namespace gpu {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:OpenCLAlteraPlugin::OpenCLAlteraPlugin() :
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:Plugin("GPU-OpenCL-Altera", "GPU-OpenCL-Altera")
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:        GPUInterface gpu;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:        bool anyGPUSupportsOpenCL = false;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:        bool anyGPUSupportsDP = false;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:        if (gpu.Initialize()) {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:            int gpuDeviceCount = gpu.GetDeviceCount();
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:            anyGPUSupportsOpenCL = (gpuDeviceCount > 0);
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:            for (int i = 0; i < gpuDeviceCount; i++) {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                gpu.GetDeviceName(i, dName, nameDescSize);
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                gpu.GetDeviceDescription(i, dDesc);
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                                        BEAGLE_FLAG_FRAMEWORK_OPENCL;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                long deviceTypeFlag = gpu.GetDeviceTypeFlag(i);
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                if (gpu.GetSupportsDoublePrecision(i)) {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                	anyGPUSupportsDP = true;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_OPENCL;
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:        if (anyGPUSupportsOpenCL) {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:            beagleFactories.push_back(new BeagleGPUImplFactory<float>());
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:            if (anyGPUSupportsDP) {
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:                beagleFactories.push_back(new BeagleGPUImplFactory<double>());
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:OpenCLAlteraPlugin::~OpenCLAlteraPlugin()
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:}	// namespace gpu
libhmsbeagle/GPU/OpenCLAlteraPlugin.cpp:	return new beagle::gpu::OpenCLAlteraPlugin();
libhmsbeagle/GPU/KernelResource.cpp://#ifdef CUDA
libhmsbeagle/GPU/KernelResource.cpp://    using namespace cuda_device;
libhmsbeagle/GPU/KernelResource.cpp://#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelResource.cpp://    using namespace opencl_device;
libhmsbeagle/GPU/GPUInterface.h:#ifndef __GPUInterface__
libhmsbeagle/GPU/GPUInterface.h:#define __GPUInterface__
libhmsbeagle/GPU/GPUInterface.h:#include "libhmsbeagle/GPU/GPUImplHelper.h"
libhmsbeagle/GPU/GPUInterface.h:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/GPUInterface.h:#include "libhmsbeagle/GPU/KernelResource.h"
libhmsbeagle/GPU/GPUInterface.h:#ifdef CUDA
libhmsbeagle/GPU/GPUInterface.h:    #include <cuda.h>
libhmsbeagle/GPU/GPUInterface.h:        #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.h"
libhmsbeagle/GPU/GPUInterface.h:        #include "libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.h"
libhmsbeagle/GPU/GPUInterface.h:    typedef CUdeviceptr GPUPtr;
libhmsbeagle/GPU/GPUInterface.h:    typedef CUfunction GPUFunction;
libhmsbeagle/GPU/GPUInterface.h:    namespace cuda_device {
libhmsbeagle/GPU/GPUInterface.h:#ifdef FW_OPENCL
libhmsbeagle/GPU/GPUInterface.h:    #define CL_USE_DEPRECATED_OPENCL_1_1_APIS // to disable deprecation warnings
libhmsbeagle/GPU/GPUInterface.h:    #define CL_USE_DEPRECATED_OPENCL_1_2_APIS // to disable deprecation warnings
libhmsbeagle/GPU/GPUInterface.h:    #define CL_USE_DEPRECATED_OPENCL_2_0_APIS // to disable deprecation warnings
libhmsbeagle/GPU/GPUInterface.h:        #include <OpenCL/opencl.h>
libhmsbeagle/GPU/GPUInterface.h:        #include <CL/opencl.h>
libhmsbeagle/GPU/GPUInterface.h:        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels_xcode.h"
libhmsbeagle/GPU/GPUInterface.h:        #include "libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h"
libhmsbeagle/GPU/GPUInterface.h:    typedef cl_mem GPUPtr;
libhmsbeagle/GPU/GPUInterface.h:    typedef cl_kernel GPUFunction;
libhmsbeagle/GPU/GPUInterface.h:    namespace opencl_device {
libhmsbeagle/GPU/GPUInterface.h:class GPUInterface {
libhmsbeagle/GPU/GPUInterface.h:#ifdef CUDA
libhmsbeagle/GPU/GPUInterface.h:    CUdevice cudaDevice;
libhmsbeagle/GPU/GPUInterface.h:    CUcontext cudaContext;
libhmsbeagle/GPU/GPUInterface.h:    CUmodule cudaModule;
libhmsbeagle/GPU/GPUInterface.h:    CUstream* cudaStreams;
libhmsbeagle/GPU/GPUInterface.h:    CUevent cudaEvent;
libhmsbeagle/GPU/GPUInterface.h:    const char* GetCUDAErrorDescription(int errorCode);
libhmsbeagle/GPU/GPUInterface.h:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/GPUInterface.h:    cl_device_id openClDeviceId;             // compute device id
libhmsbeagle/GPU/GPUInterface.h:    cl_context openClContext;                // compute context
libhmsbeagle/GPU/GPUInterface.h:    cl_command_queue* openClCommandQueues;   // compute command queue
libhmsbeagle/GPU/GPUInterface.h:    cl_event* openClEvents;                  // compute events
libhmsbeagle/GPU/GPUInterface.h:    cl_program openClProgram;                // compute program
libhmsbeagle/GPU/GPUInterface.h:    std::map<int, cl_device_id> openClDeviceMap;
libhmsbeagle/GPU/GPUInterface.h:    GPUInterface();
libhmsbeagle/GPU/GPUInterface.h:    ~GPUInterface();
libhmsbeagle/GPU/GPUInterface.h:    GPUFunction GetFunction(const char* functionName);
libhmsbeagle/GPU/GPUInterface.h:    void LaunchKernel(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterface.h:    void LaunchKernelConcurrent(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterface.h:#ifdef FW_OPENCL
libhmsbeagle/GPU/GPUInterface.h:    void* MapMemory(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterface.h:    void UnmapMemory(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterface.h:    GPUPtr AllocateMemory(size_t memSize);
libhmsbeagle/GPU/GPUInterface.h:    GPUPtr AllocateRealMemory(size_t length);
libhmsbeagle/GPU/GPUInterface.h:    GPUPtr AllocateIntMemory(size_t length);
libhmsbeagle/GPU/GPUInterface.h:    GPUPtr CreateSubPointer(GPUPtr dPtr, size_t offset, size_t size);
libhmsbeagle/GPU/GPUInterface.h:    void MemsetShort(GPUPtr dest,
libhmsbeagle/GPU/GPUInterface.h:    void MemcpyHostToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterface.h:                            const GPUPtr src,
libhmsbeagle/GPU/GPUInterface.h:    void MemcpyDeviceToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterface.h:                              GPUPtr src,
libhmsbeagle/GPU/GPUInterface.h:    void FreeMemory(GPUPtr dPtr);
libhmsbeagle/GPU/GPUInterface.h:    GPUPtr GetDeviceHostPointer(void* hPtr);
libhmsbeagle/GPU/GPUInterface.h:    void PrintfDeviceVector(GPUPtr dPtr, int length, Real r) {
libhmsbeagle/GPU/GPUInterface.h:    void PrintfDeviceVector(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterface.h:    void PrintfDeviceVector(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterface.h:    void PrintfDeviceInt(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterface.h:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/GPUInterface.h:#endif // __GPUInterface__
libhmsbeagle/GPU/GPUImplHelper.cpp:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ${CUDA_INCLUDE_DIRS}
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:add_definitions(-DCUDA)
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:     ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:     ${CUDA_TOOLKIT_ROOT_DIR}/lib64
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:    set(NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets")
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:    set(CMD_NAME createCUDAKernels.bat)
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:    set(NVCC "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc")
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:    set(NVCCFLAGS "-O3 -Wno-deprecated-gpu-targets -ccbin ${CMAKE_CXX_COMPILER} -m64 -D_POSIX_C_SOURCE -std=c++11")
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:     set(CMD_NAME make_cuda_kernels.sh)
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        COMMENT "Building CUDA kernels"
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:add_custom_target(CudaKernels DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../kernels/BeagleCUDA_kernels.h)
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:add_library(hmsbeagle-cuda SHARED
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../BeagleGPUImpl.h
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../BeagleGPUImpl.hpp
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../GPUImplDefs.h
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../GPUImplHelper.cpp
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../GPUImplHelper.h
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../GPUInterfaceCUDA.cpp
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../CUDAPlugin.cpp
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:        ../CUDAPlugin.h
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:add_dependencies(hmsbeagle-cuda CudaKernels)
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:target_link_libraries(hmsbeagle-cuda cuda ${CUDA_LIBRARIES})
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:install(TARGETS hmsbeagle-cuda
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:    COMPONENT cuda
libhmsbeagle/GPU/CMake_CUDA/CMakeLists.txt:SET_TARGET_PROPERTIES(hmsbeagle-cuda
libhmsbeagle/GPU/CMakeLists.txt:if(BUILD_OPENCL)
libhmsbeagle/GPU/CMakeLists.txt:	find_package(OpenCL)
libhmsbeagle/GPU/CMakeLists.txt:	if(OpenCL_FOUND)
libhmsbeagle/GPU/CMakeLists.txt:		message(STATUS "OpenCL Includes: ${OpenCL_INCLUDE_DIRS}")
libhmsbeagle/GPU/CMakeLists.txt:		message(STATUS "OpenCL Libraries: ${OpenCL_LIBRARIES}")
libhmsbeagle/GPU/CMakeLists.txt:		add_subdirectory("CMake_OpenCL")
libhmsbeagle/GPU/CMakeLists.txt:	endif(OpenCL_FOUND)
libhmsbeagle/GPU/CMakeLists.txt:endif(BUILD_OPENCL)
libhmsbeagle/GPU/CMakeLists.txt:if(BUILD_CUDA)
libhmsbeagle/GPU/CMakeLists.txt:	find_package(CUDA)
libhmsbeagle/GPU/CMakeLists.txt:	if(CUDA_FOUND)
libhmsbeagle/GPU/CMakeLists.txt:		message(STATUS "CUDA Includes: ${CUDA_INCLUDE_DIRS}")
libhmsbeagle/GPU/CMakeLists.txt:		add_subdirectory("CMake_CUDA")
libhmsbeagle/GPU/CMakeLists.txt:	endif(CUDA_FOUND)
libhmsbeagle/GPU/CMakeLists.txt:endif(BUILD_CUDA)
libhmsbeagle/GPU/GPUImplDefs.h:#ifndef __GPUImplDefs__
libhmsbeagle/GPU/GPUImplDefs.h:#define __GPUImplDefs__
libhmsbeagle/GPU/GPUImplDefs.h:#ifndef OPENCL_KERNEL_BUILD
libhmsbeagle/GPU/GPUImplDefs.h:#endif // OPENCL_KERNEL_BUILD
libhmsbeagle/GPU/GPUImplDefs.h://#define FW_OPENCL_BINARY
libhmsbeagle/GPU/GPUImplDefs.h://#define FW_OPENCL_TESTING
libhmsbeagle/GPU/GPUImplDefs.h://#define FW_OPENCL_PROFILING
libhmsbeagle/GPU/GPUImplDefs.h:// #define BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_GENERIC         = 0,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_INTEL_CPU       = 1,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_INTEL_GPU       = 2,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_INTEL_MIC       = 3,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_AMD_CPU         = 4,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_AMD_GPU         = 5,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_APPLE_CPU       = 6,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU   = 7,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU = 8,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_OPENCL_DEVICE_NVIDA_GPU       = 10,
libhmsbeagle/GPU/GPUImplDefs.h:    BEAGLE_CUDA_DEVICE_NVIDIA_GPU        = 11,
libhmsbeagle/GPU/GPUImplDefs.h:#ifdef CUDA
libhmsbeagle/GPU/GPUImplDefs.h:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/GPUImplDefs.h:    #define BEAGLE_STREAM_COUNT 1 // disabled for now, also has to be smaller for OpenCL to not run out of host memory
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_48_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_48_AMDGPU   4
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_48_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_64_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_64_AMDGPU   4
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_64_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_80_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_80_AMDGPU   2
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_80_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_128_AMDGPU 2
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_128_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_128_AMDGPU 2
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_192_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_192_AMDGPU  1
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_192_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_SP_256_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_SP_256_AMDGPU  1
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_SP_256_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_48_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_48_AMDGPU   4
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_48_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_64_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_64_AMDGPU   4
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_64_AMDGPU  4
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_80_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_80_AMDGPU   2
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_80_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_128_AMDGPU 2
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_128_AMDGPU  2
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_128_AMDGPU 2
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_192_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_192_AMDGPU  1
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_192_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define PATTERN_BLOCK_SIZE_DP_256_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#define MATRIX_BLOCK_SIZE_DP_256_AMDGPU  1
libhmsbeagle/GPU/GPUImplDefs.h:#define BLOCK_PEELING_SIZE_DP_256_AMDGPU 1
libhmsbeagle/GPU/GPUImplDefs.h:#if defined(FW_OPENCL_APPLECPU) && (STATE_COUNT == 4)
libhmsbeagle/GPU/GPUImplDefs.h:#elif defined(FW_OPENCL_CPU) && (STATE_COUNT == 4)
libhmsbeagle/GPU/GPUImplDefs.h:#elif (defined(FW_OPENCL_AMDGPU) || defined(FW_OPENCL_INTELGPU)) && (STATE_COUNT > 32)
libhmsbeagle/GPU/GPUImplDefs.h:    #define PATTERN_BLOCK_SIZE     GET4_VALUE(PATTERN_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
libhmsbeagle/GPU/GPUImplDefs.h:#if (defined(FW_OPENCL_AMDGPU) || defined(FW_OPENCL_INTELGPU)) && (STATE_COUNT > 32)
libhmsbeagle/GPU/GPUImplDefs.h:    #define MATRIX_BLOCK_SIZE       GET4_VALUE(MATRIX_BLOCK_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
libhmsbeagle/GPU/GPUImplDefs.h:    #define BLOCK_PEELING_SIZE      GET4_VALUE(BLOCK_PEELING_SIZE, PREC, PADDED_STATE_COUNT, AMDGPU)
libhmsbeagle/GPU/GPUImplDefs.h:#if defined(FW_OPENCL_APPLECPU)
libhmsbeagle/GPU/GPUImplDefs.h:#endif // __GPUImplDefs__
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#include "libhmsbeagle/GPU/GPUImplHelper.h"
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#include "libhmsbeagle/GPU/GPUInterface.h"
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#include "libhmsbeagle/GPU/KernelResource.h"
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                                    "\nOpenCL error: %s from file <%s>, line %i.\n", \
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:namespace opencl_device {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUInterface::GPUInterface() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClDeviceId = NULL;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClContext = NULL;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClCommandQueues = NULL;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClProgram = NULL;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUInterface::~GPUInterface() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::~GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (openClProgram != NULL)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clReleaseProgram(openClProgram));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (openClCommandQueues != NULL) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            SAFE_CL(clReleaseCommandQueue(openClCommandQueues[i]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        free(openClCommandQueues);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (openClContext != NULL)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clReleaseContext(openClContext));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::~GPUInterface\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:int GPUInterface::Initialize() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::Initialize\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            openClDeviceId = deviceIds[j];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            if (deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_CPU &&
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                //deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU &&
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                //deviceCode != BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU &&
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                //deviceCode != BEAGLE_OPENCL_DEVICE_NVIDA_GPU &&
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                openClDeviceMap.insert(std::pair<int, cl_device_id>(deviceAdded++, deviceIds[j]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            openClDeviceId = NULL;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    printf("OpenCL devices: %lu\n", openClDeviceMap.size());
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    for (int i=0; i<openClDeviceMap.size(); i++) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_NAME, param_size, param_value, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VERSION, param_size, param_value, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VENDOR, param_size, param_value, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_VENDOR_ID, sizeof(param_value_uint), &param_value_uint, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_GROUP_SIZE,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_PLATFORM,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        printf("\tOpenCL platform: ");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::Initialize\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    return (openClDeviceMap.size() ? 1 : 0);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:int GPUInterface::GetDeviceCount() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GetDeviceCount\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    for (int i=0; i<openClDeviceMap.size(); i++) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            SAFE_CL(clGetDeviceInfo(openClDeviceMap[i], CL_DEVICE_PARTITION_MAX_SUB_DEVICES, sizeof(param_value_uint), &param_value_uint, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            return openClDeviceMap.size() + param_value_uint-1;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceCount\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    return openClDeviceMap.size();
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::InitializeKernelResource(int paddedStateCount,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::InitializeKernelResource\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:               deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:               deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  -48: LOAD_KERNEL_INTO_RESOURCE( 48, DP,  48,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  -64: LOAD_KERNEL_INTO_RESOURCE( 64, DP,  64,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  -80: LOAD_KERNEL_INTO_RESOURCE( 80, DP,  80,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case -128: LOAD_KERNEL_INTO_RESOURCE(128, DP, 128,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case -192: LOAD_KERNEL_INTO_RESOURCE(192, DP, 192,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case -256: LOAD_KERNEL_INTO_RESOURCE(256, DP, 256,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case   48: LOAD_KERNEL_INTO_RESOURCE( 48, SP,  48,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case   64: LOAD_KERNEL_INTO_RESOURCE( 64, SP,  64,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case   80: LOAD_KERNEL_INTO_RESOURCE( 80, SP,  80,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  128: LOAD_KERNEL_INTO_RESOURCE(128, SP, 128,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  192: LOAD_KERNEL_INTO_RESOURCE(192, SP, 192,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            case  256: LOAD_KERNEL_INTO_RESOURCE(256, SP, 256,_AMDGPU,_AMDGPU,); break;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::InitializeKernelResource\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::SetDevice(int deviceNumber,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SetDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClDeviceId = openClDeviceMap[deviceNumber];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClContext = clCreateContext(NULL, 1, &openClDeviceId, NULL, NULL, &err);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClCommandQueues = (cl_command_queue*) malloc(sizeof(cl_command_queue) * BEAGLE_STREAM_COUNT);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClEvents = (cl_event*) malloc(sizeof(cl_event) * BEAGLE_STREAM_COUNT);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        openClCommandQueues[i] = clCreateCommandQueue(openClContext, openClDeviceId,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        openClEvents[i] = clCreateUserEvent(openClContext, &err);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#if defined(FW_OPENCL_BINARY) || defined(FW_OPENCL_PROFILING)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    #if defined(FW_OPENCL_BINARY)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    #else // FW_OPENCL_PROFILING
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    #if defined(FW_OPENCL_BINARY)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        openClProgram = clCreateProgramWithBinary(openClContext, 1, &openClDeviceId, &kernels_length,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    #else // FW_OPENCL_PROFILING
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:	    openClProgram = clCreateProgramWithSource(openClContext, 1, (const char **)&kernels,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:	openClProgram = clCreateProgramWithSource(openClContext, 1,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (!openClProgram) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        fprintf(stderr, "OpenCL error: Failed to create kernels\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    char buildDefs[1024] = "-w -D FW_OPENCL -D OPENCL_KERNEL_BUILD ";
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#elif defined(FW_OPENCL_PROFILING)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        strcat(buildDefs, "-D FW_OPENCL_CPU");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        strcat(buildDefs, "-D FW_OPENCL_CPU -D FW_OPENCL_APPLECPU");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_AMD_GPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        strcat(buildDefs, "-D FW_OPENCL_AMDGPU");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        strcat(buildDefs, "-D FW_OPENCL_AMDGPU -D FW_OPENCL_APPLEAMDGPU");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    }  else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        strcat(buildDefs, "-D FW_OPENCL_INTELGPU -D FW_OPENCL_APPLEINTELGPU");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    err = clBuildProgram(openClProgram, 0, NULL, buildDefs, NULL, NULL);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        fprintf(stderr, "OpenCL error: Failed to build kernels\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        clGetProgramBuildInfo(openClProgram, openClDeviceId, CL_PROGRAM_BUILD_LOG,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:// TODO unloading compiler to free resources is causing seg fault for Intel and NVIDIA platforms
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://     SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SetDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::ResizeStreamCount(int newStreamCount) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::ResizeStreamCount\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    // TODO: write function if using more than one opencl queue
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::ResizeStreamCount\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::SynchronizeHost() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //     SAFE_CL(clFinish(openClCommandQueues[i]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clFinish(openClCommandQueues[0]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::SynchronizeDevice() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    // SAFE_CL(clFinish(openClCommandQueues[0]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::SynchronizeDeviceWithIndex(int streamRecordIndex, int streamWaitIndex) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::SynchronizeDeviceWithIndex\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    // SAFE_CL(clFinish(openClCommandQueues[0]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::SynchronizeDeviceWithIndex\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUFunction GPUInterface::GetFunction(const char* functionName) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::GetFunction\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUFunction openClFunction;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    openClFunction = clCreateKernel(openClProgram, functionName, &err);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (!openClFunction) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        fprintf(stderr, "OpenCL error: Failed to create compute kernel %s\n", functionName);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetFunction\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    return openClFunction;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::LaunchKernel(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        void* param = (void*)(size_t)va_arg(parameters, GPUPtr);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 1, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 2, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, 3, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::LaunchKernelConcurrent(GPUFunction deviceFunction,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        void* param = (void*)(size_t)va_arg(parameters, GPUPtr);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    clGetKernelWorkGroupInfo(deviceFunction, openClDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //     //     SAFE_CL(clFinish(openClCommandQueues[streamIndex % BEAGLE_STREAM_COUNT]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //     commandQueue = openClCommandQueues[streamIndexMod];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //         // SAFE_CL(clFinish(openClCommandQueues[waitIndexMod]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //         SAFE_CL(clEnqueueBarrierWithWaitList(commandQueue, 1, &openClEvents[waitIndexMod], NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //         // SAFE_CL(clWaitForEvents(1, &openClEvents[waitIndexMod]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //         SAFE_CL(clFinish(commandQueue)); // unclear why this is needed for OpenCL CPU
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //                                    0, NULL, &openClEvents[streamIndexMod]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    //     // SAFE_CL(clEnqueueBarrierWithWaitList(commandQueue, 0, NULL, &openClEvents[streamIndexMod]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clEnqueueNDRangeKernel(openClCommandQueues[0], deviceFunction, dims, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::LaunchKernel\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void* GPUInterface::MallocHost(size_t memSize) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::MallocHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MallocHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void* GPUInterface::CallocHost(size_t size, size_t length) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::CallocHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CallocHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void* GPUInterface::AllocatePinnedHostMemory(size_t memSize, bool writeCombined, bool mapped) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:   fprintf(stderr,"\t\t\tEntering GPUInterface::AllocatePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    void* deviceBuffer = (void*) clCreateBuffer(openClContext, flags, memSize, NULL, &err);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:   fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocatePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void* GPUInterface::MapMemory(GPUPtr dPtr, size_t memSize) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    void* hostPtr = clEnqueueMapBuffer(openClCommandQueues[0], dPtr, CL_TRUE,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::UnmapMemory(GPUPtr dPtr, void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::UnmapMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clEnqueueUnmapMemObject(openClCommandQueues[0], dPtr, hPtr, 0, NULL, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving GPUInterface::UnmapMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUPtr GPUInterface::AllocateMemory(size_t memSize) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUPtr data;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, memSize, NULL, &err);
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUPtr GPUInterface::AllocateRealMemory(size_t length) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tEntering GPUInterface::AllocateRealMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUPtr data;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_REAL * length, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateRealMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUPtr GPUInterface::AllocateIntMemory(size_t length) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::AllocateIntMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUPtr data;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    data = clCreateBuffer(openClContext, CL_MEM_READ_WRITE, SIZE_INT * length, NULL,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AllocateIntMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUPtr GPUInterface::CreateSubPointer(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::CreateSubPointer\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#ifdef FW_OPENCL_ALTERA
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUPtr subPtr = dPtr;// + offset;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    GPUPtr subPtr;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if ((strcmp(param_value, "NVIDIA Corporation") != 0 && strcmp(param_value, "Apple") != 0) || offset != 0) { //TODO: use the right platform + device check
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::CreateSubPointer\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:size_t GPUInterface::AlignMemOffset(size_t offset) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::AlignMemOffset\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if ((strcmp(param_value, "NVIDIA Corporation") != 0 && strcmp(param_value, "Apple") != 0)) { //TODO: use the right platform + device check
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clGetDeviceInfo(openClDeviceId, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), &baseAlign, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::AlignMemOffset\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::MemsetShort(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://    fprintf(stderr, "\t\t\tEntering GPUInterface::MemsetShort\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemsetShort\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::MemcpyHostToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyHostToDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clEnqueueWriteBuffer(openClCommandQueues[0], dest, CL_TRUE, 0, memSize, src, 0,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyHostToDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::MemcpyDeviceToHost(void* dest,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                                      const GPUPtr src,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clEnqueueReadBuffer(openClCommandQueues[0], src, CL_TRUE, 0, memSize, dest, 0,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToHost\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::MemcpyDeviceToDevice(GPUPtr dest,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                                        GPUPtr src,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::MemcpyDeviceToDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clEnqueueCopyBuffer(openClCommandQueues[0], src, dest, 0, 0, memSize, 0,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::MemcpyDeviceToDevice\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::FreeHostMemory(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::FreePinnedHostMemory(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:   fprintf(stderr, "\t\t\tEntering GPUInterface::FreePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clReleaseMemObject((GPUPtr) hPtr));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:   fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreePinnedHostMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::FreeMemory(GPUPtr dPtr) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::FreeMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr,"\t\t\tLeaving  GPUInterface::FreeMemory\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:GPUPtr GPUInterface::GetDeviceHostPointer(void* hPtr) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceHostPointer\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://    GPUPtr dPtr;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp://    fprintf(stderr,"\t\t\tLeaving  GPUInterface::GetDeviceHostPointer\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:size_t GPUInterface::GetAvailableMemory() {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::GetDeviceName(int deviceNumber,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceName\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_NAME, sizeof(char) * nameLength, deviceName, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#if defined(BEAGLE_DEBUG_OPENCL_CORES)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_MAX_COMPUTE_UNITS,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_VERSION, param_size, param_value, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceName\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:bool GPUInterface::GetSupportsDoublePrecision(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &supportsDouble, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::GetDeviceDescription(int deviceNumber,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceDescription\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    cl_device_id tmpOpenClDevice = openClDeviceMap[deviceNumber];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong),
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_CLOCK_FREQUENCY,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(tmpOpenClDevice, CL_DEVICE_MAX_COMPUTE_UNITS,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceDescription\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::PrintfDeviceInt(GPUPtr dPtr,
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:long GPUInterface::GetDeviceTypeFlag(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceTypeFlag\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceId = openClDeviceId;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceId = openClDeviceMap[deviceNumber];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (deviceType == CL_DEVICE_TYPE_GPU)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceTypeFlag = BEAGLE_FLAG_PROCESSOR_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceTypeFlag\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:BeagleDeviceImplementationCodes GPUInterface::GetDeviceImplementationCode(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tEntering GPUInterface::GetDeviceImplementationCode\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    BeagleDeviceImplementationCodes deviceCode = BEAGLE_OPENCL_DEVICE_GENERIC;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceId = openClDeviceId;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceId = openClDeviceMap[deviceNumber];
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_CPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        else if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_INTEL_MIC;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_AMD_CPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        else if (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU)
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_AMD_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_CPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                 (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU))
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_AMD_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                 (deviceTypeFlag == BEAGLE_FLAG_PROCESSOR_GPU))
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            deviceCode = BEAGLE_OPENCL_DEVICE_APPLE_INTEL_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    } else if (!strncmp("NVIDIA", platform_string, strlen("NVIDIA"))) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        deviceCode = BEAGLE_OPENCL_DEVICE_NVIDA_GPU;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    fprintf(stderr, "\t\t\tLeaving  GPUInterface::GetDeviceImplementationCode\n");
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:const char* GPUInterface::GetCLErrorDescription(int errorCode) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#ifndef FW_OPENCL_ALTERA
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::CreateDevice(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    if (deviceNumber >= openClDeviceMap.size()) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        int coreCount = deviceNumber - openClDeviceMap.size() + 1;
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        for (int i=0; i<openClDeviceMap.size(); i++) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:            if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                SAFE_CL(clCreateSubDevices(openClDeviceMap[i],
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:                openClDeviceMap.insert(std::pair<int, cl_device_id>(openClDeviceMap.size()+coreCount-1, subdevice_id));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:void GPUInterface::ReleaseDevice(int deviceNumber) {
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:    SAFE_CL(clGetDeviceInfo(openClDeviceMap[deviceNumber], CL_DEVICE_PARENT_DEVICE, param_size, &parent_device, NULL));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        SAFE_CL(clReleaseDevice(openClDeviceMap[deviceNumber]));
libhmsbeagle/GPU/GPUInterfaceOpenCL.cpp:        openClDeviceMap.erase(deviceNumber);
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsXDerivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:# rules for building opencl files
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:#BeagleOpenCL_kernels.h: Makefile kernels4.cu kernelsX.cu kernelsAll.cu ../GPUImplDefs.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "// auto-generated header file with OpenCL kernels code" > BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "\"" >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "Making OpenCL SP state count = $s" ; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "\"" >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "#define STATE_COUNT 4" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernels4.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:cat $srcdir/kernels4Derivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:echo "\"" >> BeagleOpenCL_kernels.h
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "Making OpenCL DP state count = $s DP"; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "#define STATE_COUNT $s" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "#define DOUBLE_PRECISION" | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/../GPUImplDefs.h | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsAll.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsX.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   cat $srcdir/kernelsXDerivatives.cu | sed 's/\\/\\\\/g' | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/make_opencl_kernels.sh:   echo "\"" >> BeagleOpenCL_kernels.h; \
libhmsbeagle/GPU/kernels/createCUDAKernels.bat::: windows script to create cuda files for each state count
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:nvcc.exe -o tmp_kernels_sp_4.ptx -ptx -arch compute_60 -I../../.. -DCUDA -O3 -m64 tmp_kernels_sp_4.cu
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:nvcc.exe -o tmp_kernels_sp_%%G.ptx -ptx -arch compute_60 -I../../.. -DCUDA -O3 -m64 tmp_kernels_sp_%%G.cu
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:nvcc.exe -o tmp_kernels_dp_4.ptx -ptx -arch compute_60 -I../../.. -DCUDA -O3 -m64 tmp_kernels_dp_4.cu
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:nvcc.exe -o tmp_kernels_dp_%%G.ptx -ptx -arch compute_60 -I../../.. -DCUDA -O3 -m64 tmp_kernels_dp_%%G.cu
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:set OUTFILE="BeagleCUDA_kernels.h"
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:echo // auto-generated header file with CUDA kernels code > %OUTFILE%
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:echo #ifndef __BeagleCUDA_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:echo #define __BeagleCUDA_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/createCUDAKernels.bat:echo #endif 	// __BeagleCUDA_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat::: windows script to create a single header with OpenCL 
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:cd ..\..\..\libhmsbeagle\GPU\kernels
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels4.cl
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels%%G.cl
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_4.cl
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:type ..\GPUImplDefs.h >> kernels_dp_%%G.cl
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:set OUTFILE="BeagleOpenCL_kernels.h"
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:echo // auto-generated header file with OpenCL kernels code > %OUTFILE%
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:echo #ifndef __BeagleOpenCL_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:echo #define __BeagleOpenCL_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/createOpenCLHeader.bat:echo #endif 	// __BeagleOpenCL_kernels__ >> %OUTFILE%
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#define DETERMINE_INDICES_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#define DETERMINE_INDICES_4_MULTI_1_GPU()\
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#define DETERMINE_INDICES_4_MULTI_2_GPU()\
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:  #ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:  #else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4Derivatives.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:    #include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/kernels/kernels4.cu:    #include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels
libhmsbeagle/GPU/kernels/kernels4.cu:        partials3[deltaPartials + 0] = 1.0;/* unrolled to work around Apple OpenCL bug*/\
libhmsbeagle/GPU/kernels/kernels4.cu:// kernel macros GPU
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INDICES_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INDICES_4_MULTI_1_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INDICES_4_MULTI_2_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INDICES_4_PART_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INDICES_4_EDGEPART_1_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_PARTIALS_PARTIALS_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_PARTIALS_SINGLE_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_SCALING_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_SCALING_4_MULTI_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_SCALING_4_PART_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_MATRIX_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_MATRIX_4_MULTI_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_MATRIX_SINGLE_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define LOAD_MATRIX_DERIV_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_PARTIALS_PARTIALS_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_STATES_PARTIALS_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_STATES_STATES_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_STATES_STATES_4_SCALE_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_PARTIALS_SINGLE_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_STATES_SINGLE_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_PARTIALS_DERIV_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SUM_STATES_DERIV_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_SCALING_INDICES_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_SCALING_INDICES_4_PARTITION_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define FIND_MAX_PARTIALS_STATE_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define FIND_MAX_PARTIALS_MATRIX_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define SCALE_PARTIALS_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INTEGRATE_INDICES_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define DETERMINE_INTEGRATE_INDICES_4_PARTITION_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define INTEGRATE_PARTIALS_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#define INTEGRATE_PARTIALS_DERIV_4_GPU()\
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu://#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu://#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:////    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:////    LOAD_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu://#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU()
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_MULTI_2_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_MULTI_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_SCALE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_SCALE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_SCALING_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_STATES_4_SCALE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        #ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:    SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            #ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:        SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            #ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:    SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_SCALING_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    FIND_MAX_PARTIALS_STATE_4_DECLARE_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        FIND_MAX_PARTIALS_STATE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:            FIND_MAX_PARTIALS_MATRIX_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:                #ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:        SCALE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_PARTITION_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INTEGRATE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_EDGEPART_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_PARTIALS_4_GPU()
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_PARTIALS_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_EDGEPART_1_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_MULTI_PART_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_PARTIALS_SINGLE_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    LOAD_MATRIX_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:        SUM_STATES_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernels4.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernels4.cu:    INTEGRATE_PARTIALS_DERIV_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernels4.cu:        DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:    DETERMINE_INDICES_4_GPU();
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:#endif // CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernels4.cu:#endif //CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:    #include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:        #pragma OPENCL EXTENSION cl_khr_fp64: enable
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif //FW_OPENCL
libhmsbeagle/GPU/kernels/kernelsAll.cu:#if (defined CUDA) && (defined DOUBLE_PRECISION) &&  (__CUDA_ARCH__ < 600)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_AMDGPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_AMDGPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#if !(defined(FW_OPENCL_APPLEAMDGPU) && defined(DOUBLE_PRECISION)) // TODO: fix this issue
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#elif defined(FW_OPENCL)
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_AMDGPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_AMDGPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:// #ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsAll.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsAll.cu:#endif //CUDA
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:# For OpenCL, we need to generate the file `BeagleOpenCL_kernels.h` using the commands (and dependencies) below
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:# rules for building cuda files
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:#BeagleCUDA_kernels.h: Makefile kernels4.cu kernels4Derivatives.cu kernelsX.cu kernelsXDerivatives.cu kernelsAll.cu ../GPUImplDefs.h
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	echo "// auto-generated header file with CUDA kernels PTX code" > BeagleCUDA_kernels.h
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	${NVCC} -o BeagleCUDA_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		$srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDA_kernels.h; exit; }; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	echo "#define KERNELS_STRING_SP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	echo "\"" >> BeagleCUDA_kernels.h
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "Making CUDA SP state count = $s" ; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		${NVCC} -o BeagleCUDA_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:			$srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDA_kernels.h; exit; }; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "#define KERNELS_STRING_SP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "\"" >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	${NVCC} -o BeagleCUDA_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=4 -DDOUBLE_PRECISION \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		$srcdir/kernels4.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDA_kernels.h; exit; }; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	echo "#define KERNELS_STRING_DP_4 \"" | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:	echo "\"" >> BeagleCUDA_kernels.h
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "Making CUDA DP state count = $s" ; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		${NVCC} -o BeagleCUDA_kernels.ptx --default-stream per-thread -ptx -DCUDA -DSTATE_COUNT=$s -DDOUBLE_PRECISION \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:			$srcdir/kernelsX.cu ${NVCCFLAGS} -DHAVE_CONFIG_H ${INCLUDE_DIRS} || { \rm BeagleCUDA_kernels.h; exit; }; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "#define KERNELS_STRING_DP_$s \"" | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		cat BeagleCUDA_kernels.ptx | sed 's/\"/\\"/g' | sed 's/$/\\n\\/' >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/make_cuda_kernels.sh:		echo "\"" >> BeagleCUDA_kernels.h; \
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:    #include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/kernels/kernelsX.cu:    #include "libhmsbeagle/GPU/kernels/kernelsAll.cu" // This file includes the non-state-count specific kernels
libhmsbeagle/GPU/kernels/kernelsX.cu:// kernel macros GPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#define DETERMINE_INDICES_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define LOAD_SCALING_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_PARTIALS_PARTIALS_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_STATES_PARTIALS_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define LOAD_PARTIALS_SCALING_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define FIND_MAX_PARTIALS_STATE_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define FIND_MAX_PARTIALS_MATRIX_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SCALE_PARTIALS_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define INTEGRATE_PARTIALS_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define INTEGRATE_PARTIALS_DERIV_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_STATES_POWER_OF_TWO_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_STATES_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu:#define SUM_STATES_DERIVS_X_GPU()\
libhmsbeagle/GPU/kernels/kernelsX.cu://#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu://#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:////    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_PARTIALS_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_PARTIALS_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_PARTIALS_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        FIND_MAX_PARTIALS_MATRIX_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SCALE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_PARTIALS_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        FIND_MAX_PARTIALS_MATRIX_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SCALE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_PARTIALS_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        FIND_MAX_PARTIALS_MATRIX_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        #ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:    SCALE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    LOAD_PARTIALS_SCALING_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    FIND_MAX_PARTIALS_STATE_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        FIND_MAX_PARTIALS_MATRIX_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:            #ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:    SCALE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:        SUM_STATES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_DERIV_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_DERIVS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef FW_OPENCL_CPU // CPU/MIC implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:#else // GPU implementation
libhmsbeagle/GPU/kernels/kernelsX.cu:    INTEGRATE_PARTIALS_DERIV_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_DERIVS_POWER_OF_TWO_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:    SUM_STATES_DERIVS_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // FW_OPENCL_CPU
libhmsbeagle/GPU/kernels/kernelsX.cu:    DETERMINE_INDICES_X_GPU();
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif // CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:#ifdef CUDA
libhmsbeagle/GPU/kernels/kernelsX.cu:#endif //CUDA
libhmsbeagle/GPU/OpenCLPlugin.h:#ifndef __BEAGLE_OPENCL_PLUGIN_H__
libhmsbeagle/GPU/OpenCLPlugin.h:#define __BEAGLE_OPENCL_PLUGIN_H__
libhmsbeagle/GPU/OpenCLPlugin.h:namespace gpu {
libhmsbeagle/GPU/OpenCLPlugin.h:class BEAGLE_DLLEXPORT OpenCLPlugin : public beagle::plugin::Plugin
libhmsbeagle/GPU/OpenCLPlugin.h:	OpenCLPlugin();
libhmsbeagle/GPU/OpenCLPlugin.h:	~OpenCLPlugin();
libhmsbeagle/GPU/OpenCLPlugin.h:	OpenCLPlugin( const OpenCLPlugin& cp );	// disallow copy by defining this private
libhmsbeagle/GPU/OpenCLPlugin.h:} // namespace gpu
libhmsbeagle/GPU/OpenCLPlugin.h:#endif	// __BEAGLE_OPENCL_PLUGIN_H__
libhmsbeagle/GPU/CUDAPlugin.cpp:#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
libhmsbeagle/GPU/CUDAPlugin.cpp:#include "libhmsbeagle/GPU/CUDAPlugin.h"
libhmsbeagle/GPU/CUDAPlugin.cpp:namespace gpu {
libhmsbeagle/GPU/CUDAPlugin.cpp:CUDAPlugin::CUDAPlugin() :
libhmsbeagle/GPU/CUDAPlugin.cpp:Plugin("GPU-CUDA", "GPU-CUDA")
libhmsbeagle/GPU/CUDAPlugin.cpp:        GPUInterface gpu;
libhmsbeagle/GPU/CUDAPlugin.cpp:        bool anyGPUSupportsCUDA = false;
libhmsbeagle/GPU/CUDAPlugin.cpp:        bool anyGPUSupportsDP = false;
libhmsbeagle/GPU/CUDAPlugin.cpp:        if (gpu.Initialize()) {
libhmsbeagle/GPU/CUDAPlugin.cpp:            int gpuDeviceCount = gpu.GetDeviceCount();
libhmsbeagle/GPU/CUDAPlugin.cpp:            anyGPUSupportsCUDA = (gpuDeviceCount > 0);
libhmsbeagle/GPU/CUDAPlugin.cpp:            for (int i = 0; i < gpuDeviceCount; i++) {
libhmsbeagle/GPU/CUDAPlugin.cpp:                gpu.GetDeviceName(i, dName, nameDescSize);
libhmsbeagle/GPU/CUDAPlugin.cpp:                gpu.GetDeviceDescription(i, dDesc);
libhmsbeagle/GPU/CUDAPlugin.cpp:                                        BEAGLE_FLAG_PROCESSOR_GPU |
libhmsbeagle/GPU/CUDAPlugin.cpp:                                        BEAGLE_FLAG_FRAMEWORK_CUDA;
libhmsbeagle/GPU/CUDAPlugin.cpp:                if (gpu.GetSupportsDoublePrecision(i)) {
libhmsbeagle/GPU/CUDAPlugin.cpp:                	anyGPUSupportsDP = true;
libhmsbeagle/GPU/CUDAPlugin.cpp:                resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_CUDA;
libhmsbeagle/GPU/CUDAPlugin.cpp:    if (anyGPUSupportsCUDA) {
libhmsbeagle/GPU/CUDAPlugin.cpp:    	using namespace cuda;
libhmsbeagle/GPU/CUDAPlugin.cpp:        if (anyGPUSupportsDP) {
libhmsbeagle/GPU/CUDAPlugin.cpp:            beagleFactories.push_back(new BeagleGPUImplFactory<double>());
libhmsbeagle/GPU/CUDAPlugin.cpp:        beagleFactories.push_back(new BeagleGPUImplFactory<float>());
libhmsbeagle/GPU/CUDAPlugin.cpp:CUDAPlugin::~CUDAPlugin()
libhmsbeagle/GPU/CUDAPlugin.cpp:}	// namespace gpu
libhmsbeagle/GPU/CUDAPlugin.cpp:	return new beagle::gpu::CUDAPlugin();
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:#ifndef __BEAGLE_OPENCL_ALTERA_PLUGIN_H__
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:#define __BEAGLE_OPENCL_ALTERA_PLUGIN_H__
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:namespace gpu {
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:class BEAGLE_DLLEXPORT OpenCLAlteraPlugin : public beagle::plugin::Plugin
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:	OpenCLAlteraPlugin();
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:	~OpenCLAlteraPlugin();
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:	OpenCLAlteraPlugin( const OpenCLAlteraPlugin& cp );	// disallow copy by defining this private
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:} // namespace gpu
libhmsbeagle/GPU/OpenCLAlteraPlugin.h:#endif	// __BEAGLE_OPENCL_ALTERA_PLUGIN_H__
libhmsbeagle/GPU/OpenCLPlugin.cpp:#include "libhmsbeagle/GPU/BeagleGPUImpl.h"
libhmsbeagle/GPU/OpenCLPlugin.cpp:#include "libhmsbeagle/GPU/OpenCLPlugin.h"
libhmsbeagle/GPU/OpenCLPlugin.cpp:namespace gpu {
libhmsbeagle/GPU/OpenCLPlugin.cpp:OpenCLPlugin::OpenCLPlugin() :
libhmsbeagle/GPU/OpenCLPlugin.cpp:Plugin("GPU-OpenCL", "GPU-OpenCL")
libhmsbeagle/GPU/OpenCLPlugin.cpp:        GPUInterface gpu;
libhmsbeagle/GPU/OpenCLPlugin.cpp:        bool anyGPUSupportsOpenCL = false;
libhmsbeagle/GPU/OpenCLPlugin.cpp:        bool anyGPUSupportsDP = false;
libhmsbeagle/GPU/OpenCLPlugin.cpp:        if (gpu.Initialize()) {
libhmsbeagle/GPU/OpenCLPlugin.cpp:            int gpuDeviceCount = gpu.GetDeviceCount();
libhmsbeagle/GPU/OpenCLPlugin.cpp:            anyGPUSupportsOpenCL = (gpuDeviceCount > 0);
libhmsbeagle/GPU/OpenCLPlugin.cpp:            for (int i = 0; i < gpuDeviceCount; i++) {
libhmsbeagle/GPU/OpenCLPlugin.cpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/OpenCLPlugin.cpp:                gpu.CreateDevice(i);
libhmsbeagle/GPU/OpenCLPlugin.cpp:                gpu.GetDeviceName(i, dName, nameDescSize);
libhmsbeagle/GPU/OpenCLPlugin.cpp:                gpu.GetDeviceDescription(i, dDesc);
libhmsbeagle/GPU/OpenCLPlugin.cpp:                                        BEAGLE_FLAG_FRAMEWORK_OPENCL;
libhmsbeagle/GPU/OpenCLPlugin.cpp:                long deviceTypeFlag = gpu.GetDeviceTypeFlag(i);
libhmsbeagle/GPU/OpenCLPlugin.cpp:                if (gpu.GetSupportsDoublePrecision(i)) {
libhmsbeagle/GPU/OpenCLPlugin.cpp:                	anyGPUSupportsDP = true;
libhmsbeagle/GPU/OpenCLPlugin.cpp:                resource.requiredFlags = BEAGLE_FLAG_FRAMEWORK_OPENCL;
libhmsbeagle/GPU/OpenCLPlugin.cpp:#ifdef BEAGLE_DEBUG_OPENCL_CORES
libhmsbeagle/GPU/OpenCLPlugin.cpp:                gpu.ReleaseDevice(i);
libhmsbeagle/GPU/OpenCLPlugin.cpp:        if (anyGPUSupportsOpenCL) {
libhmsbeagle/GPU/OpenCLPlugin.cpp:        	using namespace opencl;
libhmsbeagle/GPU/OpenCLPlugin.cpp:            if (anyGPUSupportsDP) {
libhmsbeagle/GPU/OpenCLPlugin.cpp:                beagleFactories.push_back(new BeagleGPUImplFactory<double>());
libhmsbeagle/GPU/OpenCLPlugin.cpp:            beagleFactories.push_back(new BeagleGPUImplFactory<float>());
libhmsbeagle/GPU/OpenCLPlugin.cpp:OpenCLPlugin::~OpenCLPlugin()
libhmsbeagle/GPU/OpenCLPlugin.cpp:}	// namespace gpu
libhmsbeagle/GPU/OpenCLPlugin.cpp:	return new beagle::gpu::OpenCLPlugin();
libhmsbeagle/GPU/KernelLauncher.cpp:#include "libhmsbeagle/GPU/GPUImplDefs.h"
libhmsbeagle/GPU/KernelLauncher.cpp:#include "libhmsbeagle/GPU/KernelLauncher.h"
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef CUDA
libhmsbeagle/GPU/KernelLauncher.cpp:namespace cuda_device {
libhmsbeagle/GPU/KernelLauncher.cpp:namespace opencl_device {
libhmsbeagle/GPU/KernelLauncher.cpp:KernelLauncher::KernelLauncher(GPUInterface* inGpu) {
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu = inGpu;
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    BeagleDeviceImplementationCodes deviceCode = gpu->GetDeviceImplementationCode(-1);
libhmsbeagle/GPU/KernelLauncher.cpp:    if (deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_CPU ||
libhmsbeagle/GPU/KernelLauncher.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_INTEL_MIC ||
libhmsbeagle/GPU/KernelLauncher.cpp:        deviceCode == BEAGLE_OPENCL_DEVICE_AMD_CPU) {
libhmsbeagle/GPU/KernelLauncher.cpp:    } else if (deviceCode == BEAGLE_OPENCL_DEVICE_APPLE_CPU) {
libhmsbeagle/GPU/KernelLauncher.cpp:    kPaddedStateCount = gpu->kernelResource->paddedStateCount;
libhmsbeagle/GPU/KernelLauncher.cpp:    kCategoryCount = gpu->kernelResource->categoryCount;
libhmsbeagle/GPU/KernelLauncher.cpp:    kPatternCount = gpu->kernelResource->patternCount;
libhmsbeagle/GPU/KernelLauncher.cpp:    kUnpaddedPatternCount = gpu->kernelResource->unpaddedPatternCount;
libhmsbeagle/GPU/KernelLauncher.cpp:    kMultiplyBlockSize = gpu->kernelResource->multiplyBlockSize;
libhmsbeagle/GPU/KernelLauncher.cpp:    kPatternBlockSize = gpu->kernelResource->patternBlockSize;
libhmsbeagle/GPU/KernelLauncher.cpp:    kSlowReweighing = gpu->kernelResource->slowReweighing;
libhmsbeagle/GPU/KernelLauncher.cpp:    kMatrixBlockSize = gpu->kernelResource->matrixBlockSize;
libhmsbeagle/GPU/KernelLauncher.cpp:    kFlags = gpu->kernelResource->flags;
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL_TESTING
libhmsbeagle/GPU/KernelLauncher.cpp:	fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fIntegrateLikelihoods = gpu->GetFunction("kernelIntegrateLikelihoods");
libhmsbeagle/GPU/KernelLauncher.cpp:    fSumSites1 = gpu->GetFunction("kernelSumSites1");
libhmsbeagle/GPU/KernelLauncher.cpp:	fMatrixConvolution = gpu ->GetFunction("kernelMatrixConvolution");
libhmsbeagle/GPU/KernelLauncher.cpp:	fMatrixTranspose = gpu->GetFunction("kernelMatrixTranspose");
libhmsbeagle/GPU/KernelLauncher.cpp:        fMatrixMulADBMulti = gpu->GetFunction("kernelMatrixMulADBComplexMulti");
libhmsbeagle/GPU/KernelLauncher.cpp:        fMatrixMulADBMulti = gpu->GetFunction("kernelMatrixMulADBMulti");
libhmsbeagle/GPU/KernelLauncher.cpp:    fMatrixMulADBFirstDeriv = gpu->GetFunction("kernelMatrixMulADBFirstDeriv");
libhmsbeagle/GPU/KernelLauncher.cpp:    fMatrixMulADBSecondDeriv = gpu->GetFunction("kernelMatrixMulADBSecondDeriv");
libhmsbeagle/GPU/KernelLauncher.cpp:		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADBComplex");
libhmsbeagle/GPU/KernelLauncher.cpp:		fMatrixMulADB = gpu->GetFunction("kernelMatrixMulADB");
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockCoherent = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockFixedScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockAutoScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsGrowing = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsStatesGrowing = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsEdgeFirstDerivatives = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsStatesEdgeFirstDerivatives = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fMultipleNodeSiteReduction = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fMultipleNodeSiteSquaredReduction = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsCrossProducts = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsStatesCrossProducts = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockCheckScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsByPatternBlockFixedCheckScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fStatesPartialsByPatternBlockCoherent = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fStatesStatesByPatternBlockCoherent = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsByPatternBlockFixedScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesStatesByPatternBlockFixedScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsEdgeLikelihoods = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsPartialsEdgeLikelihoodsSecondDeriv = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fStatesPartialsEdgeLikelihoods = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fStatesPartialsEdgeLikelihoodsSecondDeriv = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fIntegrateLikelihoodsDynamicScalingSecondDeriv = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction("kernelIntegrateLikelihoodsAutoScaling");
libhmsbeagle/GPU/KernelLauncher.cpp:        fIntegrateLikelihoodsDynamicScaling = gpu->GetFunction("kernelIntegrateLikelihoodsFixedScale");
libhmsbeagle/GPU/KernelLauncher.cpp:        fAccumulateFactorsDynamicScaling = gpu->GetFunction("kernelAccumulateFactorsScalersLog");
libhmsbeagle/GPU/KernelLauncher.cpp:        fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactorsScalersLog");
libhmsbeagle/GPU/KernelLauncher.cpp:        fAccumulateFactorsDynamicScaling = gpu->GetFunction("kernelAccumulateFactors");
libhmsbeagle/GPU/KernelLauncher.cpp:        fRemoveFactorsDynamicScaling = gpu->GetFunction("kernelRemoveFactors");
libhmsbeagle/GPU/KernelLauncher.cpp:    fAccumulateFactorsAutoScaling = gpu->GetFunction("kernelAccumulateFactorsAutoScaling");
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScaling = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fPartialsDynamicScalingAccumulate = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsDynamicScalingAccumulateDifference = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fPartialsDynamicScalingAccumulateReciprocal = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:    fIntegrateLikelihoods = gpu->GetFunction("kernelIntegrateLikelihoods");
libhmsbeagle/GPU/KernelLauncher.cpp:    fIntegrateLikelihoodsSecondDeriv = gpu->GetFunction("kernelIntegrateLikelihoodsSecondDeriv");
libhmsbeagle/GPU/KernelLauncher.cpp:    fIntegrateLikelihoodsMulti = gpu->GetFunction("kernelIntegrateLikelihoodsMulti");
libhmsbeagle/GPU/KernelLauncher.cpp:	fIntegrateLikelihoodsFixedScaleMulti = gpu->GetFunction("kernelIntegrateLikelihoodsFixedScaleMulti");
libhmsbeagle/GPU/KernelLauncher.cpp:    fSumSites1 = gpu->GetFunction("kernelSumSites1");
libhmsbeagle/GPU/KernelLauncher.cpp:    fSumSites2 = gpu->GetFunction("kernelSumSites2");
libhmsbeagle/GPU/KernelLauncher.cpp:    fSumSites3 = gpu->GetFunction("kernelSumSites3");
libhmsbeagle/GPU/KernelLauncher.cpp:    fReorderPatterns = gpu->GetFunction("kernelReorderPatterns");
libhmsbeagle/GPU/KernelLauncher.cpp:        fPartialsPartialsByPatternBlockCoherentMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fPartialsPartialsByPatternBlockCoherentPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fPartialsPartialsByPatternBlockFixedScalingMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fPartialsPartialsByPatternBlockFixedScalingPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsByPatternBlockCoherentMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsByPatternBlockCoherentPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesStatesByPatternBlockCoherentMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesStatesByPatternBlockCoherentPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsByPatternBlockFixedScalingMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsByPatternBlockFixedScalingPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesStatesByPatternBlockFixedScalingMulti = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesStatesByPatternBlockFixedScalingPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fPartialsPartialsEdgeLikelihoodsByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fStatesPartialsEdgeLikelihoodsByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fIntegrateLikelihoodsDynamicScalingPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fResetFactorsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fAccumulateFactorsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fRemoveFactorsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fRemoveFactorsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:            fAccumulateFactorsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:                fPartialsDynamicScalingAccumulateByPartition = gpu->GetFunction(
libhmsbeagle/GPU/KernelLauncher.cpp:        fIntegrateLikelihoodsPartition = gpu->GetFunction("kernelIntegrateLikelihoodsPartition");
libhmsbeagle/GPU/KernelLauncher.cpp:        fSumSites1Partition = gpu->GetFunction("kernelSumSites1Partition");
libhmsbeagle/GPU/KernelLauncher.cpp:#endif // !FW_OPENCL_TESTING
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::ReorderPatterns(GPUPtr dPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dStates,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dStatesSort,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dTipOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dTipTypes,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dPatternsNewOrder,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr dPatternWeightsSort,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fReorderPatterns,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::ConvolveTransitionMatrices(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:		                                        GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:	gpu->LaunchKernel(fMatrixConvolution, bgTransitionProbabilitiesBlock,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::TransposeTransitionMatrices(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:		                                         GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:	gpu->LaunchKernel(fMatrixTranspose, bgTransitionProbabilitiesBlock,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::GetTransitionProbabilitiesSquareMulti(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMatrixMulADBMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::GetTransitionProbabilitiesSquare(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMatrixMulADB,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::GetTransitionProbabilitiesSquareFirstDeriv(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMatrixMulADBFirstDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::GetTransitionProbabilitiesSquareSecondDeriv(GPUPtr dMatrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dEvec,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dIevc,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr dEigenValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr distanceQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMatrixMulADBSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsStatesEdgeFirstDerivatives(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr states0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr partials0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr matrices0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsStatesEdgeFirstDerivatives,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsEdgeFirstDerivatives(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                          GPUPtr partials0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                          GPUPtr matrices0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                          GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                          GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsEdgeFirstDerivatives,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsStatesCrossProducts(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                 GPUPtr states0,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr lengths,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr categoryWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr patternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsStatesCrossProducts,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsCrossProducts(GPUPtr out,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr lengths,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr instructions,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr categoryWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr patternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsCrossProducts,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::MultipleNodeSiteReduction(GPUPtr outSiteValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                               GPUPtr inSiteValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                               GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMultipleNodeSiteReduction,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::MultipleNodeSiteSquaredReduction(GPUPtr outSiteValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr inSiteValues,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                      GPUPtr weights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fMultipleNodeSiteSquaredReduction,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsStatesGrowing(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                           GPUPtr states2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                           GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                           GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                           GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsStatesGrowing,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsGrowing(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                             GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                             GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                             GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                             GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsGrowing,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsPruningDynamicCheckScaling(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr* dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr* dScalingFactorsMaster,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr dRescalingTrigger,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fPartialsPartialsByPatternBlockCheckScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:                dScalingFactors[writeScalingIndex] = gpu->AllocateMemory(patternCount * sizeReal);
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeReal *patternCount);
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernel(fPartialsDynamicScalingAccumulateReciprocal,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fPartialsPartialsByPatternBlockFixedCheckScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:                dScalingFactors[writeScalingIndex] = gpu->AllocateRealMemory(patternCount);
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->MemcpyDeviceToDevice(dScalingFactorsMaster[cumulativeScalingIndex], dScalingFactors[cumulativeScalingIndex], sizeReal * patternCount);
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->SynchronizeDevice();
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernel(fPartialsDynamicScalingAccumulateDifference,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsPruningMulti(GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                  GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                  GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                  GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fPartialsPartialsByPatternBlockCoherentMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fPartialsPartialsByPatternBlockFixedScalingMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsPruningDynamicScaling(GPUPtr partials1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                           GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fPartialsPartialsByPatternBlockAutoScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockCoherentPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockCoherent,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockFixedScalingPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fPartialsPartialsByPatternBlockFixedScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesPartialsPruningMulti(GPUPtr states,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fStatesPartialsByPatternBlockCoherentMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fStatesPartialsByPatternBlockFixedScalingMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesPartialsPruningDynamicScaling(GPUPtr states1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr partials2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherentPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherent,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherentPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockCoherent,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockFixedScalingPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesPartialsByPatternBlockFixedScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // restore values if used fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesStatesPruningMulti(GPUPtr states,
libhmsbeagle/GPU/KernelLauncher.cpp:                                              GPUPtr partials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                              GPUPtr matrices,
libhmsbeagle/GPU/KernelLauncher.cpp:                                              GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                              GPUPtr ptrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fStatesStatesByPatternBlockCoherentMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernel(fStatesStatesByPatternBlockFixedScalingMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesStatesPruningDynamicScaling(GPUPtr states1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr states2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr matrices1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr matrices2,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                       GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherentPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:            gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherent,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherentPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockCoherent,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockFixedScalingPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:                gpu->LaunchKernelConcurrent(fStatesStatesByPatternBlockFixedScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // restore values if used fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsDynamicScaling(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                        GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsDynamicScalingPartition(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                 GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScalingPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsDynamicScalingSecondDeriv(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dFirstDerivResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dSecondDerivResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dRootFirstDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dRootSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                   GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsDynamicScalingSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dChildParials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoods,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dPartialsOrigin,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dMatricesOrigin,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoodsByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::PartialsPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dFirstDerivTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dSecondDerivTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dChildParials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dFirstDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dSecondDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fPartialsPartialsEdgeLikelihoodsSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesPartialsEdgeLikelihoods(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dChildStates,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoods,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // restore values if used fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesPartialsEdgeLikelihoodsByPartition(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                              GPUPtr dPartialsOrigin,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                              GPUPtr dStatesOrigin,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                              GPUPtr dMatricesOrigin,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                              GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoodsByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::StatesPartialsEdgeLikelihoodsSecondDeriv(GPUPtr dPartialsTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dFirstDerivTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dSecondDerivTmp,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dParentPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dChildStates,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dTransMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dFirstDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dSecondDerivMatrix,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fStatesPartialsEdgeLikelihoodsSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:#ifdef FW_OPENCL
libhmsbeagle/GPU/KernelLauncher.cpp:    // restore values if used fix for Apple CPU OpenCL limitations
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::AccumulateFactorsDynamicScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fAccumulateFactorsDynamicScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::AccumulateFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                                GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fAccumulateFactorsDynamicScalingByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::AccumulateFactorsAutoScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                  GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                  GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fAccumulateFactorsAutoScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::RemoveFactorsDynamicScaling(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                 GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                     GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fRemoveFactorsDynamicScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::RemoveFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                            GPUPtr dNodePtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                            GPUPtr dRootScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fRemoveFactorsDynamicScalingByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::ResetFactorsDynamicScalingByPartition(GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fResetFactorsDynamicScalingByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::RescalePartials(GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                     GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.cpp://            gpu->MemcpyHostToDevice(scalingFactors, ones, SIZE_REAL * patternCount);
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingAccumulate,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernelConcurrent(fPartialsDynamicScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::RescalePartialsByPartition(GPUPtr partials3,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr scalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                GPUPtr cumulativeScaling,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingAccumulateByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:        gpu->LaunchKernelConcurrent(fPartialsDynamicScalingByPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoods(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoods,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsPartition(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                   GPUPtr dPtrOffsets,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsPartition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsSecondDeriv(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dFirstDerivResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dSecondDerivResult,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dRootFirstDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dRootSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:                                          GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsSecondDeriv,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsMulti(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:											   GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:											   GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:											   GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::IntegrateLikelihoodsFixedScaleMulti(GPUPtr dResult,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dRootPartials,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dFrequencies,
libhmsbeagle/GPU/KernelLauncher.cpp:                                                         GPUPtr dScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dPtrQueue,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dMaxScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:														 GPUPtr dIndexMaxScalingFactors,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fIntegrateLikelihoodsFixedScaleMulti,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::SumSites1(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fSumSites1,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::SumSites1Partition(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                        GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.cpp:                                        GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fSumSites1Partition,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::SumSites2(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dArray2,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum2,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fSumSites2,
libhmsbeagle/GPU/KernelLauncher.cpp:void KernelLauncher::SumSites3(GPUPtr dArray1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum1,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dArray2,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum2,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dArray3,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dSum3,
libhmsbeagle/GPU/KernelLauncher.cpp:                              GPUPtr dPatternWeights,
libhmsbeagle/GPU/KernelLauncher.cpp:    gpu->LaunchKernel(fSumSites3,
libhmsbeagle/beagle.cpp:        std::cerr << "Loading hmsbeagle-cuda" << std::endl;
libhmsbeagle/beagle.cpp:        beagle::plugin::Plugin* cudaplug = pm.findPlugin("hmsbeagle-cuda");
libhmsbeagle/beagle.cpp:        plugins->push_back(cudaplug);
libhmsbeagle/beagle.cpp:        std::cerr << "Unable to load hmsbeagle-cuda: " << sle.getError() << std::endl;
libhmsbeagle/beagle.cpp:        std::cerr << "Loading hmsbeagle-opencl" << std::endl;
libhmsbeagle/beagle.cpp:        beagle::plugin::Plugin* openclplug = pm.findPlugin("hmsbeagle-opencl");
libhmsbeagle/beagle.cpp:        plugins->push_back(openclplug);
libhmsbeagle/beagle.cpp:        std::cerr << "Unable to load hmsbeagle-opencl: " << sle.getError() << std::endl;
libhmsbeagle/beagle.cpp:        beagle::plugin::Plugin* openclalteraplug = pm.findPlugin("hmsbeagle-opencl-altera");
libhmsbeagle/beagle.cpp:        plugins->push_back(openclalteraplug);
libhmsbeagle/CPU/sse2neon.h://   Brandon Rowlett <browlett@nvidia.com>
libhmsbeagle/CPU/sse2neon.h://   Eric van Beurden <evanbeurden@nvidia.com>
libhmsbeagle/CPU/sse2neon.h://   Alexander Potylitsin <apotylitsin@nvidia.com>
libhmsbeagle/CPU/sse2neon.h://   Cuda Chen <clh960524@gmail.com>
libhmsbeagle/CMakeLists.txt:add_subdirectory(GPU)
libhmsbeagle/benchmark/BeagleBenchmark.cpp://     if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
libhmsbeagle/benchmark/BeagleBenchmark.cpp://     if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
libhmsbeagle/benchmark/BeagleBenchmark.cpp://     if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/matrixtest/matrixtest.cpp:                                  BEAGLE_FLAG_PROCESSOR_GPU,             	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
examples/tinytest/tinytest.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
examples/tinytest/tinytest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
examples/tinytest/tinytest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/tinytest/tinytest.cpp:                                BEAGLE_FLAG_FRAMEWORK_CUDA |
examples/tinytest/tinytest.cpp:                                BEAGLE_FLAG_PRECISION_SINGLE | BEAGLE_FLAG_PROCESSOR_GPU | (autoScaling ? BEAGLE_FLAG_SCALING_AUTO : 0),	/**< Bit-flags indicating preferred implementation charactertistics, see BeagleFlags (input) */
examples/hmctest/hmcGapTest.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
examples/hmctest/hmcGapTest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
examples/hmctest/hmcGapTest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/hmctest/hmcGapTest.cpp:    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;
examples/hmctest/hmcGapTest.cpp:    if (useGpu) {
examples/hmctest/hmcGapTest.cpp:    if (useGpu) {
examples/hmctest/hmcGapTest.cpp:        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
examples/hmctest/hmcGapTest.cpp:    int transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
examples/hmctest/hmcWeibullTest.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
examples/hmctest/hmcWeibullTest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
examples/hmctest/hmcWeibullTest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/hmctest/hmcWeibullTest.cpp:    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;
examples/hmctest/hmcWeibullTest.cpp:    if (useGpu) {
examples/hmctest/hmcWeibullTest.cpp:    if (useGpu) {
examples/hmctest/hmcWeibullTest.cpp:        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
examples/hmctest/hmcWeibullTest.cpp:    int transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
examples/hmctest/hmctest5.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
examples/hmctest/hmctest5.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
examples/hmctest/hmctest5.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/hmctest/hmctest5.cpp:    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;
examples/hmctest/hmctest5.cpp:    if (useGpu) {
examples/hmctest/hmctest5.cpp:    if (useGpu) {
examples/hmctest/hmctest5.cpp:        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
examples/hmctest/hmctest5.cpp:    if (autoTranspose || !useGpu) {
examples/hmctest/hmctest5.cpp:        transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
examples/hmctest/hmctest.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU)      fprintf(stdout, " PROCESSOR_GPU");
examples/hmctest/hmctest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA)     fprintf(stdout, " FRAMEWORK_CUDA");
examples/hmctest/hmctest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL)   fprintf(stdout, " FRAMEWORK_OPENCL");
examples/hmctest/hmctest.cpp:    bool useGpu = argc > 1 && strcmp(argv[1] , "--gpu") == 0;
examples/hmctest/hmctest.cpp:    if (useGpu) {
examples/hmctest/hmctest.cpp:    if (useGpu) {
examples/hmctest/hmctest.cpp:        preferenceFlags |= BEAGLE_FLAG_PROCESSOR_GPU;
examples/hmctest/hmctest.cpp:    int transpose = (stateCount == 4 || !useGpu) ? 0 : 6;
examples/synthetictest/synthetictest.cpp:    if (inFlags & BEAGLE_FLAG_PROCESSOR_GPU      ) fprintf(stdout, " PROCESSOR_GPU"      );
examples/synthetictest/synthetictest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_CUDA     ) fprintf(stdout, " FRAMEWORK_CUDA"     );
examples/synthetictest/synthetictest.cpp:    if (inFlags & BEAGLE_FLAG_FRAMEWORK_OPENCL   ) fprintf(stdout, " FRAMEWORK_OPENCL"   );
examples/synthetictest/synthetictest.cpp:               bool opencl,
examples/synthetictest/synthetictest.cpp:                    (opencl ? BEAGLE_FLAG_FRAMEWORK_OPENCL : 0) |
examples/synthetictest/synthetictest.cpp:    std::cerr << "synthetictest [--help] [--resourcelist] [--benchmarklist] [--states <integer>] [--taxa <integer>] [--sites <integer>] [--rates <integer>] [--manualscale] [--autoscale] [--dynamicscale] [--rsrc <integer>] [--reps <integer>] [--doubleprecision] [--disablevector] [--enablethreads] [--compacttips <integer>] [--seed <integer>] [--rescalefrequency <integer>] [--fulltiming] [--unrooted] [--calcderivs] [--logscalers] [--eigencount <integer>] [--eigencomplex] [--ievectrans] [--setmatrix] [--opencl] [--partitions <integer>] [--sitelikes] [--newdata] [--randomtree] [--reroot] [--stdrand] [--pectinate] [--multirsrc] [--postorder] [--newtree] [--newparameters] [--threadcount] [--clientthreads]";
examples/synthetictest/synthetictest.cpp:                                    bool* opencl,
examples/synthetictest/synthetictest.cpp:        } else if (option == "--opencl") {
examples/synthetictest/synthetictest.cpp:            *opencl = true;
examples/synthetictest/synthetictest.cpp:    bool opencl = false;
examples/synthetictest/synthetictest.cpp:                                   &eigenCount, &eigencomplex, &ievectrans, &setmatrix, &opencl,
examples/synthetictest/synthetictest.cpp:                          opencl,
examples/standalone/epochtest/configure.ac:AC_INIT([EpochTest C++], [0.1], [bug-report@hello.beagle-gpu.com],
examples/standalone/epochtest/configure.ac:             [hellobeagle], [http://hello.beagle-gpu.com/])
examples/standalone/hellobeagle/configure.ac:AC_INIT([HelloBeagle C++], [0.1], [bug-report@hello.beagle-gpu.com],
examples/standalone/hellobeagle/configure.ac:             [hellobeagle], [http://hello.beagle-gpu.com/])
examples/swig_python/beagle.i:    BEAGLE_FLAG_PROCESSOR_GPU       = 1 << 16,   /**< Use GPU as main processor */
examples/swig_python/beagle.i:    BEAGLE_FLAG_FRAMEWORK_CUDA      = 1 << 22,   /**< Use CUDA implementation with GPU resources */
examples/swig_python/beagle.i:    BEAGLE_FLAG_FRAMEWORK_OPENCL    = 1 << 23    /**< Use OpenCL implementation with GPU resources */
examples/swig_python/beagle_wrap.c:  SWIG_Python_SetConstant(d, "BEAGLE_FLAG_PROCESSOR_GPU",SWIG_From_int((int)(BEAGLE_FLAG_PROCESSOR_GPU)));
examples/swig_python/beagle_wrap.c:  SWIG_Python_SetConstant(d, "BEAGLE_FLAG_FRAMEWORK_CUDA",SWIG_From_int((int)(BEAGLE_FLAG_FRAMEWORK_CUDA)));
examples/swig_python/beagle_wrap.c:  SWIG_Python_SetConstant(d, "BEAGLE_FLAG_FRAMEWORK_OPENCL",SWIG_From_int((int)(BEAGLE_FLAG_FRAMEWORK_OPENCL)));
examples/swig_python/beagle.py:BEAGLE_FLAG_PROCESSOR_GPU = _beagle.BEAGLE_FLAG_PROCESSOR_GPU
examples/swig_python/beagle.py:BEAGLE_FLAG_FRAMEWORK_CUDA = _beagle.BEAGLE_FLAG_FRAMEWORK_CUDA
examples/swig_python/beagle.py:BEAGLE_FLAG_FRAMEWORK_OPENCL = _beagle.BEAGLE_FLAG_FRAMEWORK_OPENCL
.travis.yml:            - nvidia-opencl-dev
.travis.yml:#    - ${OPENCL_ROOT}
.travis.yml:      mkdir -p ${OPENCL_ROOT}
.travis.yml:      export OPENCL_VENDOR_PATH=${AMDAPPSDKROOT}/etc/OpenCL/vendors
.travis.yml:      mkdir -p ${OPENCL_VENDOR_PATH}
.travis.yml:      echo libamdocl64.so > ${OPENCL_VENDOR_PATH}/amdocl64.icd
.travis.yml:  - OPENCL_ROOT=$HOME/opencl
.travis.yml:  - OPENCL_LIB=amdappsdk
.travis.yml:  - OPENCL_VERSION="12"
.travis.yml:  - AMDAPPSDK_VERSION=291 # OpenCL 1.2
.travis.yml:  - AMDAPPSDKROOT=${OPENCL_ROOT}/AMDAPPSDK
.gitignore:# /libhmsbeagle/GPU/
.gitignore:/libhmsbeagle/GPU/Makefile.in
.gitignore:/libhmsbeagle/GPU/PeelingFunctions.linkinfo
.gitignore:/libhmsbeagle/GPU/.deps
.gitignore:/libhmsbeagle/GPU/TransitionFunctions.linkinfo
.gitignore:/libhmsbeagle/GPU/Makefile
.gitignore:/libhmsbeagle/GPU/BeagleCUDA_kernels.h
.gitignore:/libhmsbeagle/GPU/BeagleCUDA_kernels.ptx
.gitignore:# /libhmsbeagle/GPU/kernels/
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_64.ptx
.gitignore:/libhmsbeagle/GPU/kernels/Makefile.in
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_48.ptx
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.h
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels.ptx
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_4.ptx
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_32.ptx
.gitignore:/libhmsbeagle/GPU/kernels/Makefile
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.h
.gitignore:/libhmsbeagle/GPU/kernels/BeagleCUDA_kernels_xcode.ptx
.gitignore:libhmsbeagle/GPU/.libs/
.gitignore:libhmsbeagle/GPU/libhmsbeagle-cuda.la
.gitignore:libhmsbeagle/GPU/libhmsbeagle_cuda_la-CUDAPlugin.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_cuda_la-GPUImplHelper.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_cuda_la-GPUInterfaceCUDA.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_cuda_la-KernelLauncher.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_cuda_la-KernelResource.lo
.gitignore:libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels.h
.gitignore:libhmsbeagle/GPU/kernels/BeagleOpenCL_kernels_xcode.h
.gitignore:libhmsbeagle/GPU/libhmsbeagle-opencl.la
.gitignore:libhmsbeagle/GPU/libhmsbeagle_opencl_la-GPUImplHelper.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_opencl_la-GPUInterfaceOpenCL.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_opencl_la-KernelLauncher.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_opencl_la-KernelResource.lo
.gitignore:libhmsbeagle/GPU/libhmsbeagle_opencl_la-OpenCLPlugin.lo

```
