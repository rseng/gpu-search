# https://github.com/fmihpc/vlasiator

```console
vlasovsolver/cpu_trans_map_amr.cpp:      // CUDATODO: use blockGID to get pointers here
doxytmp/Doxyfile:OUTPUT_DIRECTORY       = /home/sandroos/codes/cuda/FVM2/doc
doxytmp/Doxyfile:INPUT                  = /home/sandroos/codes/cuda/FVM2
MAKE/Makefile.mahti_gcc:# also load cuda/11.5.0 for nvtx support
fieldtracing/fieldtracing.h:      bool doTraceOpenClosed=false;
fieldtracing/fieldtracing.h:   void traceOpenClosedConnection(
fieldtracing/fieldtracing.cpp:      if(fieldTracingParameters.doTraceOpenClosed) {
fieldtracing/fieldtracing.cpp:         traceOpenClosedConnection(technicalGrid, perBGrid, dPerBGrid, nodes);
fieldtracing/fieldtracing.cpp:   void traceOpenClosedConnection(
fieldtracing/fieldtracing.cpp:      phiprof::Timer tracingTimer {"fieldtracing-ionosphere-openclosedTracing"};
doc/vlsv2/vlsv2_structure.eps:%%Title: /home/sandroos/codes/cuda/cudafvm/doc/vlsv2/vlsv2_structure.dia
doc/fieldsolver/boundaries.eps:%%Title: /home/sandroos/codes/cuda/cudafvm/doc/fieldsolver/boundaries.dia
doc/fieldsolver/variables.eps:%%Title: /home/sandroos/codes/cuda/cudafvm/doc/fieldsolver/variables.dia
doc/fieldsolver/Hall_term/2013_summer/variables.eps:%%Title: /home/sandroos/codes/cuda/cudafvm/doc/fieldsolver/variables.dia
datareduction/datareducer.cpp:      if(P::systemWriteAllDROs || lowercase == "ig_openclosed") {
datareduction/datareducer.cpp:         FieldTracing::fieldTracingParameters.doTraceOpenClosed = true;
datareduction/datareducer.cpp:         outputReducer->addOperator(new DRO::DataReductionOperatorIonosphereNode("ig_openclosed", [](
spatial_cell_wrapper.hpp:  Spatial cell wrapper, maps to GPU or CPU version
fieldsolver/fs_common.h:// Constants: not needed as such, but if field solver is implemented on GPUs 
fieldsolver/fs_common.h:// CPU and GPU results.
parameters.cpp:                        "ig_electrontemp ig_solverinternals ig_upmappednodecoords ig_upmappedb ig_openclosed ig_potential "+

```
