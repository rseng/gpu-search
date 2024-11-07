# https://github.com/matthiaslee/Xmatch

```console
BoostXmatch/BoostXmatch/CudaContext.h:#ifndef CUDACONTEXT_H
BoostXmatch/BoostXmatch/CudaContext.h:#define CUDACONTEXT_H
BoostXmatch/BoostXmatch/CudaContext.h:#include "CudaManager.h"
BoostXmatch/BoostXmatch/CudaContext.h:#include <cuda.h>
BoostXmatch/BoostXmatch/CudaContext.h:	class CudaContext
BoostXmatch/BoostXmatch/CudaContext.h:		CudaManagerPtr cuman;
BoostXmatch/BoostXmatch/CudaContext.h:		CudaContext(CudaManagerPtr cuman);
BoostXmatch/BoostXmatch/CudaContext.h:		~CudaContext();
BoostXmatch/BoostXmatch/CudaContext.h:	typedef boost::shared_ptr<CudaContext> CudaContextPtr;
BoostXmatch/BoostXmatch/CudaContext.h:#endif /* CUDACONTEXT_H */
BoostXmatch/BoostXmatch/CudaManager.cpp:#include "CudaManager.h"
BoostXmatch/BoostXmatch/CudaManager.cpp:#include <cuda_runtime.h>
BoostXmatch/BoostXmatch/CudaManager.cpp:#include <cuda.h>
BoostXmatch/BoostXmatch/CudaManager.cpp:	CudaManager::CudaManager() : available()
BoostXmatch/BoostXmatch/CudaManager.cpp:		if (error != CUDA_SUCCESS) nDevices = 0;
BoostXmatch/BoostXmatch/CudaManager.cpp:			available[i] = (error == CUDA_SUCCESS);
BoostXmatch/BoostXmatch/CudaManager.cpp:	int CudaManager::NextDevice()
BoostXmatch/BoostXmatch/CudaManager.cpp:	void CudaManager::Release(int id)
BoostXmatch/BoostXmatch/CudaManager.cpp:	void CudaManager::BlacklistDevice(int id)
BoostXmatch/BoostXmatch/CudaManager.cpp:		// FORCE blacklist of display GPU
BoostXmatch/BoostXmatch/CudaManager.cpp:		cudaDeviceProp prop;
BoostXmatch/BoostXmatch/CudaManager.cpp:		  cudaGetDeviceProperties(&prop,i);
BoostXmatch/BoostXmatch/CudaManager.cpp:	bool MeetsReq (const cudaDeviceProp& dev, const cudaDeviceProp& req)
BoostXmatch/BoostXmatch/CudaManager.cpp:	std::vector<int> CudaManager::Query(const cudaDeviceProp& req)
BoostXmatch/BoostXmatch/CudaManager.cpp:		cudaError_t err = cudaGetDeviceCount(&n);
BoostXmatch/BoostXmatch/CudaManager.cpp:		if (err == cudaSuccess)
BoostXmatch/BoostXmatch/CudaManager.cpp:			cudaDeviceProp prop;
BoostXmatch/BoostXmatch/CudaManager.cpp:				cudaGetDeviceProperties(&prop,i);
BoostXmatch/BoostXmatch/CudaManager.cpp:	void Print(cudaDeviceProp devProp)
BoostXmatch/BoostXmatch/Worker.cu:#include "CudaContext.h"
BoostXmatch/BoostXmatch/Worker.cu:#include <cuda.h>
BoostXmatch/BoostXmatch/Worker.cu:#include <cuda_runtime.h>
BoostXmatch/BoostXmatch/Worker.cu:#include <cuda_runtime_api.h>
BoostXmatch/BoostXmatch/Worker.cu:	// convert ra,dec to xyz (gpu w/ single point and less precise "fast math")
BoostXmatch/BoostXmatch/Worker.cu:		CUDA kernel
BoostXmatch/BoostXmatch/Worker.cu:		LOG_TIM << "- GPU-" << id << " " << *job << " copying to device" << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:		// copy to gpu
BoostXmatch/BoostXmatch/Worker.cu:			    cudaFree(oldjob->ptrA);
BoostXmatch/BoostXmatch/Worker.cu:			cudaMalloc(&p1_radec, sizeof(dbl2)*job->segA->vRadec.size());
BoostXmatch/BoostXmatch/Worker.cu:			cudaMemcpy(p1_radec, &(job->segA->vRadec[0]), job->segA->vRadec.size(), cudaMemcpyHostToDevice);
BoostXmatch/BoostXmatch/Worker.cu:			    cudaFree(oldjob->ptrB);
BoostXmatch/BoostXmatch/Worker.cu:			cudaMalloc(&p2_radec, sizeof(dbl2)*job->segB->vRadec.size());
BoostXmatch/BoostXmatch/Worker.cu:			cudaMemcpy(p2_radec, &(job->segB->vRadec[0]), job->segB->vRadec.size(), cudaMemcpyHostToDevice);
BoostXmatch/BoostXmatch/Worker.cu:		  cudaMalloc(&p1_radec, sizeof(dbl2)*job->segA->vRadec.size());
BoostXmatch/BoostXmatch/Worker.cu:		  cudaMemcpy(p1_radec, &(job->segA->vRadec[0]), job->segA->vRadec.size(), cudaMemcpyHostToDevice);
BoostXmatch/BoostXmatch/Worker.cu:		  cudaMalloc(&p2_radec, sizeof(dbl2)*job->segB->vRadec.size());
BoostXmatch/BoostXmatch/Worker.cu:		  cudaMemcpy(p2_radec, &(job->segB->vRadec[0]), job->segB->vRadec.size(), cudaMemcpyHostToDevice);
BoostXmatch/BoostXmatch/Worker.cu:		LOG_TIM << "- GPU-" << id << " " << *job << " kernel launches" << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:		cudaEvent_t start_event, stop_event;
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventCreate(&start_event);
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventCreate(&stop_event);
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventRecord(start_event, 0);
BoostXmatch/BoostXmatch/Worker.cu:		cudaDeviceSynchronize();
BoostXmatch/BoostXmatch/Worker.cu:				//cudaDeviceSynchronize();
BoostXmatch/BoostXmatch/Worker.cu:			// could cuda-sync here and dump (smaller) result sets on the fly...
BoostXmatch/BoostXmatch/Worker.cu:		LOG_TIM << "- GPU-" << id << " " << "syncing.." << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:		cudaDeviceSynchronize();
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventRecord(stop_event, 0);
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventSynchronize(stop_event);
BoostXmatch/BoostXmatch/Worker.cu:		cudaEventElapsedTime(&calc_time, start_event, stop_event);
BoostXmatch/BoostXmatch/Worker.cu:		LOG_TIM << "- GPU-" << id << " " << *job << " # " << match_num << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:			LOG_ERR << "- GPU-" << id << " " << *job << " !! Truncated output !!" << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:		  cudaFree(p1_radec);
BoostXmatch/BoostXmatch/Worker.cu:		  cudaFree(p2_radec);
BoostXmatch/BoostXmatch/Worker.cu:		LOG_PRG << "- GPU-" << id << " " << *job << " done" << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:			CudaContextPtr ctx(new CudaContext(cuman));
BoostXmatch/BoostXmatch/Worker.cu:				LOG_ERR << "- Thread-" << id << " !! Cannot get CUDA context !!" << std::endl; 
BoostXmatch/BoostXmatch/Worker.cu:				LOG_DBG << "- GPU-" << id << " output to file " << outpath << std::endl; 
BoostXmatch/BoostXmatch/Worker.cu:					LOG_ERR << "- GPU-" << id << " !! Cannot open output file !!" << std::endl;
BoostXmatch/BoostXmatch/Worker.cu:			LOG_ERR << "- GPU-" << id << " !! Error !!" << std::endl
BoostXmatch/BoostXmatch/Worker.cu:			LOG_ERR << "- GPU-" << id << " !! Unknown error !!" << std::endl;	
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <ClInclude Include="CudaManager.h">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <ClInclude Include="CudaContext.h">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <CudaCompile Include="Sorter.cu">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <CudaCompile Include="Worker.cu">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <CudaCompile Include="Common.cu">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <ClCompile Include="CudaManager.cpp">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj.filters:    <ClCompile Include="CudaContext.cpp">
BoostXmatch/BoostXmatch/CudaManager.h:#ifndef CUDAMANAGER_H
BoostXmatch/BoostXmatch/CudaManager.h:#define CUDAMANAGER_H
BoostXmatch/BoostXmatch/CudaManager.h:	class CudaManager
BoostXmatch/BoostXmatch/CudaManager.h:		CudaManager();
BoostXmatch/BoostXmatch/CudaManager.h:	typedef boost::shared_ptr<CudaManager> CudaManagerPtr;
BoostXmatch/BoostXmatch/CudaManager.h:#endif /* CUDAMANAGER_H */
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 4.1.props" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:      <AdditionalIncludeDirectories>c:\tamas\projects\boost-trunk;$(CUDA_INC_PATH)</AdditionalIncludeDirectories>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:      <AdditionalDependencies>$(CUDA_LIB_PATH)\cudart.lib;$(CUDA_LIB_PATH)\cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:      <GPUDebugInfo>true</GPUDebugInfo>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:      <AdditionalIncludeDirectories>c:\tamas\projects\boost-trunk;$(CUDA_INC_PATH)</AdditionalIncludeDirectories>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:      <AdditionalDependencies>$(CUDA_LIB_PATH)\cudart.lib;$(CUDA_LIB_PATH)\cuda.lib;%(AdditionalDependencies)</AdditionalDependencies>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <ClCompile Include="CudaContext.cpp" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <ClCompile Include="CudaManager.cpp">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <CudaCompile Include="Sorter.cu" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <CudaCompile Include="Worker.cu" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <ClInclude Include="CudaContext.h" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <ClInclude Include="CudaManager.h" />
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <CudaCompile Include="Common.cu">
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    </CudaCompile>
BoostXmatch/BoostXmatch/BoostXmatch.vcxproj:    <Import Project="$(VCTargetsPath)\BuildCustomizations\CUDA 4.1.targets" />
BoostXmatch/BoostXmatch/Sorter.h:#include "CudaManager.h"
BoostXmatch/BoostXmatch/Sorter.h:		CudaManagerPtr cuman;
BoostXmatch/BoostXmatch/Sorter.h:		Sorter(CudaManagerPtr cuman, uint32_t id, SegmentManagerPtr segman, double sr_deg, double zh_deg, int verbosity) 
BoostXmatch/BoostXmatch/BoostXmatch.cpp:#include "CudaManager.h"
BoostXmatch/BoostXmatch/BoostXmatch.cpp:					("threads,t", po::value<uint32_t>(&num_threads)->default_value(0), "number of threads, defaults to # of GPUs")
BoostXmatch/BoostXmatch/BoostXmatch.cpp:		3) Main thread: Load enough segments for GPUs from the larger file
BoostXmatch/BoostXmatch/BoostXmatch.cpp:		// cuda query
BoostXmatch/BoostXmatch/BoostXmatch.cpp:		CudaManagerPtr cuman(new CudaManager());
BoostXmatch/BoostXmatch/BoostXmatch.cpp:		LOG_DBG << "# of GPUs: " << cuman->GetDeviceCount() << std::endl;
BoostXmatch/BoostXmatch/BoostXmatch.cpp:		//Blacklist Display GPU
BoostXmatch/BoostXmatch/Makefile:#export CUDA_INCLUDE=/opt/cuda/include
BoostXmatch/BoostXmatch/Makefile:#export CUDA_LIB=/opt/cuda/lib64 
BoostXmatch/BoostXmatch/Makefile:#CUDA_INCLUDE=
BoostXmatch/BoostXmatch/Makefile:#CUDA_LIB=
BoostXmatch/BoostXmatch/Makefile:INCLUDE =	-I${CUDA_INCLUDE} -I${BOOST_DIR}
BoostXmatch/BoostXmatch/Makefile:			-L${CUDA_LIB} -lcudart -lcuda
BoostXmatch/BoostXmatch/Makefile:			CudaContext.o \
BoostXmatch/BoostXmatch/Makefile:			CudaManager.o \
BoostXmatch/BoostXmatch/Worker.h:#include "CudaManager.h"
BoostXmatch/BoostXmatch/Worker.h:		CudaManagerPtr cuman;
BoostXmatch/BoostXmatch/Worker.h:		Worker(CudaManagerPtr cuman, uint32_t id, JobManagerPtr jobman, std::string outpath, uint32_t maxout, int verbosity) 
BoostXmatch/BoostXmatch/Sorter.cu:#include "CudaContext.h"
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_TIM << "- GPU-" << id << " " << *seg <<" copying to device" << std::endl;
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_TIM << "- GPU-" << id << " " << *seg <<" sorting by zoneid, ra" << std::endl;
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_DBG << "- GPU-" << id << " " << *seg <<" zone boundaries" << std::endl;
BoostXmatch/BoostXmatch/Sorter.cu:		// zone limits on gpu
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_DBG << "- GPU-" << id << " " << *seg <<" splitting" << std::endl;
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_TIM << "- GPU-" << id << " " << *seg <<" copying to host" << std::endl;		
BoostXmatch/BoostXmatch/Sorter.cu:		LOG_TIM << "- GPU-" << id << " " << *seg <<" done" << std::endl;
BoostXmatch/BoostXmatch/Sorter.cu:			CudaContextPtr ctx(new CudaContext(cuman));
BoostXmatch/BoostXmatch/Sorter.cu:				LOG_ERR << "- Thread-" << id << " !! Cannot get CUDA context !!" << std::endl; 
BoostXmatch/BoostXmatch/Debug Switches.txt:This gives an exception sometimes for 3 GPUS when "Sorting segments" but works with 1/2 or if -v4 
BoostXmatch/BoostXmatch/CudaContext.cpp:#include "CudaContext.h"
BoostXmatch/BoostXmatch/CudaContext.cpp:	CudaContext::CudaContext(CudaManagerPtr cuman) : cuman(cuman)
BoostXmatch/BoostXmatch/CudaContext.cpp:		if( cuResult != CUDA_SUCCESS )
BoostXmatch/BoostXmatch/CudaContext.cpp:	CudaContext::~CudaContext()
README:A GPU accelerated astronomic catalog crossmatching tool.
README:- Prooven working with segement size of 50million across 4 GPUs(Tesla c2070)

```
