# https://github.com/mivp/encube

```console
control/lib/sharevol-all.js:  //Yes it's user agent sniffing, but we need to attempt to detect mobile devices so we don't over-stress their gpu...
control/src/main.js:  //Yes it's user agent sniffing, but we need to attempt to detect mobile devices so we don't over-stress their gpu...
hdsupport.h:#define GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX          0x9047
hdsupport.h:#define GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX    0x9048
hdsupport.h:#define GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX  0x9049
hdsupport.h:#define GPU_MEMORY_INFO_EVICTION_COUNT_NVX            0x904A
hdsupport.h:#define GPU_MEMORY_INFO_EVICTED_MEMORY_NVX            0x904B
hdsupport.h:void queryGPUInfo();
hdsupport.cpp:void queryGPUInfo()
hdsupport.cpp:    if(!glewIsSupported("GL_NVX_gpu_memory_info"))
hdsupport.cpp:        fprintf(stderr, "GL_NVX_gpu_memory_info NOT supported!\n");
hdsupport.cpp:    glGetIntegerv(GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &totalMem);
hdsupport.cpp:    glGetIntegerv(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avaiMem);
hdsupport.cpp:    fprintf(stderr, "GPU TOTAL MEM: %7.2fMB, GPU AVAILABLE MEM: %7.2fMB\n", totalMem/1024.0, avaiMem/1024.0);
hdsupport.cpp:        queryGPUInfo();
s2hd.c:    if(glewIsSupported("GL_NVX_gpu_memory_info"))
s2hd.c:        glGetIntegerv(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avaiMem);
encube_om.cpp:#define GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX          0x9047
encube_om.cpp:#define GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX    0x9048
encube_om.cpp:#define GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX  0x9049
encube_om.cpp:#define GPU_MEMORY_INFO_EVICTION_COUNT_NVX            0x904A
encube_om.cpp:#define GPU_MEMORY_INFO_EVICTED_MEMORY_NVX            0x904B
encube_om.cpp:  if(glewIsSupported("GL_NVX_gpu_memory_info")) {
encube_om.cpp:    glGetIntegerv(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avaiMem);

```
