# https://github.com/David-McKenna/udpPacketManager

```console
docs/COMPILERS.md:lofar_udp_extractor -i ../20230418181951B0834+06/udp_1613[[port]].ucc1.2023-04-18T18\:20\:00.000.zst -p ${procmode} -o ./debug_[[idx]] -T ${threads} | tee ${compiler}_T${threads}_${procmode}
tests/lib_tests/lib_metadata_tests.cpp:		//__attribute__((unused)) int _sigprocMachineID(const char *machineName); // Currently not implemented
paper/paper.bib:  abstract = {The LOw Frequency ARray - LOFAR - is a new radio interferometer designed with emphasis on flexible digital hardware instead of mechanical solutions. The array elements, so-called stations, are located in the Netherlands and in neighbouring countries. The design of LOFAR allows independent use of its international stations, which, coupled with a dedicated backend, makes them very powerful telescopes in their own right. This backend is called the Advanced Radio Transient Event Monitor and Identification System (ARTEMIS). It is a combined software/hardware solution for both targeted observations and real-time searches for millisecond radio transients which uses Graphical Processing Unit (GPU) technology to remove interstellar dispersion and detect millisecond radio bursts from astronomical sources in real-time.},
CMakeLists.txt:                    CONFIGURE_COMMAND ./bootstrap && ./configure --with-cuda-dir=no && cd 3rdparty && make libtimers.la # libtimers fails to compile during normal step with older Make versions
src/lib/lofar_udp_metadata.c:			if (metadata->upm_procmode != guppiMode) {
src/lib/lofar_udp_metadata.c:				fprintf(stderr, "WARNING %s: GUPPI Raw Headers are intended to be attached to mode %d data (currently in mode %d).\n", __func__, guppiMode, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:			if (metadata->upm_procmode < stokesMin) {
src/lib/lofar_udp_metadata.c:				fprintf(stderr, "WARNING %s: Sigproc headers are intended to be attached to Stokes data (mode >= %d, currently in %d).\n", __func__, stokesMin, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:			if (metadata->upm_procmode < stokesMin || metadata->upm_procmode > hdf5StokesMax) {
src/lib/lofar_udp_metadata.c:				fprintf(stderr, "ERROR %s: Requested HDF5 metadata, but processing mode is %d. The HDF5 writer only supports frequency-major data (between %d and %d), eixting.\n", __func__, metadata->upm_procmode, stokesMin, hdf5StokesMax);
src/lib/lofar_udp_metadata.c:		metadata->upm_procmode = reader->meta->processingMode;
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			fprintf(stderr,"ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:	if (metadata->upm_procmode > 99) {
src/lib/lofar_udp_metadata.c:		samplingTime *= (double) (1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes I, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "Q-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes Q, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "U-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes U, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "V-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes V, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "Q-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[2], META_STR_LEN, "U-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[3], META_STR_LEN, "V-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IQUV, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "V-POS-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IV, with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes I, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "Q-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes Q, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "U-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes U, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "V-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes V, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "Q-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[2], META_STR_LEN, "U-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[3], META_STR_LEN, "V-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IQUV, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "V-NEG-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IV, with reversed frequencies and %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes I, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "Q-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes Q, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "U-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes U, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "V-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes V, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "Q-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[2], META_STR_LEN, "U-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[3], META_STR_LEN, "V-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IQUV, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[0], META_STR_LEN, "I-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt[1], META_STR_LEN, "V-TME-%dx", 1 << metadata->upm_procmode % 10);
src/lib/lofar_udp_metadata.c:			snprintf(metadata->upm_outputfmt_comment, META_STR_LEN, "Stokes IV, time-major with %dx downsampling", 1 << (metadata->upm_procmode % 10));
src/lib/lofar_udp_metadata.c:			fprintf(stderr, "ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			fprintf(stderr, "ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			fprintf(stderr, "ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			fprintf(stderr, "ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_metadata.c:	switch (metadata->upm_procmode) {
src/lib/lofar_udp_metadata.c:			fprintf(stderr, "ERROR %s: Unknown processing mode %d, exiting.\n", __func__, metadata->upm_procmode);
src/lib/lofar_udp_structs_metadata.h:	processMode_t upm_procmode;
src/lib/lofar_udp_metadata.h:__attribute__((unused)) int32_t _sigprocMachineID(const char *machineName); // Currently not implemented
src/lib/io/lofar_udp_io_HDF5.c:				if (metadata->upm_procmode < 100) {
src/lib/io/lofar_udp_io_HDF5.c:				{ "COMPLEX_VOLTAGE", (metadata->upm_procmode < 100)},
src/lib/metadata/lofar_udp_metadata_SIGPROC.c:	//metadata->output.sigproc->machine_id = sigprocMachineID(metadata->machine);
src/lib/metadata/lofar_udp_metadata_SIGPROC.c:__attribute__((unused)) int32_t _sigprocMachineID(const char *machineName) {
src/lib/metadata/lofar_udp_metadata_DADA.c:	returnVal += _writeInt_DADA(workingBuffer, "UPM_MODE", hdr->upm_procmode);
src/lib/lofar_udp_structs_metadata.c:	.upm_procmode = UNSET_MODE,

```
