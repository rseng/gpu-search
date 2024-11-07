# https://github.com/philres/ngmlr

```console
docker/build-env/Dockerfile:# binutils is required to run opencl programs
src/SAMWriter.cpp:		char const * rgPu = Config.getRgPu();
src/SAMWriter.cpp:		if(rgPu != 0)
src/SAMWriter.cpp:			Print("\tPU:%s", rgPu);
src/IConfig.h:	char * rgPu = 0;
src/IConfig.h:	char const * const getRgPu() const {
src/IConfig.h:		return rgPu;
src/ScoreBuffer.cpp:	//Adding scores to buffer. If buffer full, submit to CPU/GPU for score computation
src/ScoreBuffer.cpp:	//Force submitting remaining computation from buffer to CPU/GPU
src/ArgParser.cpp:	TCLAP::ValueArg<std::string> rgPuArg("", "rg-pu", "RG header: Platform unit", false, noneDefault, "string", cmd);
src/ArgParser.cpp:	printParameter<std::string>(usage, rgPuArg);
src/ArgParser.cpp:	rgPu = fromString(rgPuArg.getValue());
src/IAlignment.h:typedef IAlignment * (*pfCreateAlignment)(int const gpu_id);

```
