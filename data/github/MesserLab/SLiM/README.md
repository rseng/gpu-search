# https://github.com/MesserLab/SLiM

```console
QtSLiM/QtSLiMWindow.cpp:    //                     843 gldDestroyContext  (in libGPUSupportMercury.dylib) + 114  [0x7fff57fe6745]
QtSLiM/QtSLiMIndividualsWidget.cpp:	// is probably memory-bound since most of the work is done in the GPU.  I haven't done any speed tests to
SLiMgui/PopulationView.mm:	// is probably memory-bound since most of the work is done in the GPU.  I haven't done any speed tests to
eidos/eidos_globals.cpp:		// Look for devices (GPUs, accelerators) that we are able to offload to.
eidos/eidos_globals.cpp:		// Note that OpenMP offloading to the GPUs on Apple Silicon is not currently supported by any compiler.
eidos/eidos_globals.cpp:			(*outstream) << "// ********** OpenMP target device count (GPUs, accelerators): " << num_devices << std::endl;

```
