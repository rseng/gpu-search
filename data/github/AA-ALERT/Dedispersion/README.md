# https://github.com/AA-ALERT/Dedispersion

```console
README.md:Many-core incoherent dedispersion algorithm in OpenCL, with classes to use them in C++.
README.md:* [OpenCL](https://github.com/isazi/OpenCL) - master branch
README.md:Checks if the output of the CPU is the same for the GPU.
README.md: * *opencl_platform*     OpenCL platform
README.md: * *opencl_device*       OpenCL device number
README.md: * *padding*             cacheline size, in bytes, of the OpenCL device
README.md: * *vector*              vector size, in number of input items, of the OpenCL device
README.md: *  *local*                  Defines OpenCL memmory space to use; ie. automatic or manual caching.
README.md:Classses holding the implementation of the kernels for CPU and GPU.
include/Dedispersion.hpp:#include <OpenCLTypes.hpp>
include/Dedispersion.hpp:class DedispersionConf : public isa::OpenCL::KernelConf {
include/Dedispersion.hpp:// OpenCL
include/Dedispersion.hpp:template< typename I, typename O > std::string * getDedispersionOpenCL(const DedispersionConf & conf, const unsigned int padding, const uint8_t inputBits, const std::string & inputDataType, const std::string & intermediateDataType, const std::string & outputDataType, const AstroData::Observation & observation, std::vector< float > & shifts);
include/Dedispersion.hpp:template< typename I, typename O > std::string * getSubbandDedispersionStepOneOpenCL(const DedispersionConf & conf, const unsigned int padding, const uint8_t inputBits, const std::string & inputDataType, const std::string & intermediateDataType, const std::string & outputDataType, const AstroData::Observation & observation, std::vector< float > & shifts);
include/Dedispersion.hpp:template< typename I > std::string * getSubbandDedispersionStepTwoOpenCL(const DedispersionConf & conf, const unsigned int padding, const std::string & inputDataType, const AstroData::Observation & observation, std::vector< float > & shifts);
include/Dedispersion.hpp:template< typename I, typename O > std::string * getDedispersionOpenCL(const DedispersionConf & conf, const unsigned int padding, const uint8_t inputBits, const std::string & inputDataType, const std::string & intermediateDataType, const std::string & outputDataType, const AstroData::Observation & observation, std::vector< float > & shifts)
include/Dedispersion.hpp:        unrolled_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(bit)), std::to_string(bit));
include/Dedispersion.hpp:          unrolled_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(inputBits - 1)), std::to_string(bit));
include/Dedispersion.hpp:        sum_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(bit)), std::to_string(bit));
include/Dedispersion.hpp:          sum_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(inputBits - 1)), std::to_string(bit));
include/Dedispersion.hpp:template< typename I, typename O > std::string * getSubbandDedispersionStepOneOpenCL(const DedispersionConf & conf, const unsigned int padding, const uint8_t inputBits, const std::string & inputDataType, const std::string & intermediateDataType, const std::string & outputDataType, const AstroData::Observation & observation, std::vector< float > & shifts)
include/Dedispersion.hpp:        unrolled_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(bit)), std::to_string(bit));
include/Dedispersion.hpp:          unrolled_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(inputBits - 1)), std::to_string(bit));
include/Dedispersion.hpp:        sum_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(bit)), std::to_string(bit));
include/Dedispersion.hpp:          sum_sTemplate += isa::OpenCL::setBit("interBuffer", isa::OpenCL::getBit("bitsBuffer", "firstBit + " + std::to_string(inputBits - 1)), std::to_string(bit));
include/Dedispersion.hpp:template< typename I > std::string * getSubbandDedispersionStepTwoOpenCL(const DedispersionConf & conf, const unsigned int padding, const std::string & inputDataType, const AstroData::Observation & observation, std::vector< float > & shifts)
CMakeLists.txt:set(TARGET_LINK_LIBRARIES dedispersion isa_utils isa_opencl astrodata OpenCL)
CMakeLists.txt:  set(TARGET_LINK_LIBRARIES ${TARGET_LINK_LIBRARIES} psrdada cudart)
src/Dedispersion.cpp:  return std::to_string(splitBatches) + " " + std::to_string(local) + " " + std::to_string(unroll) + " " + isa::OpenCL::KernelConf::print();
src/DedispersionTuning.cpp:#include <InitializeOpenCL.hpp>
src/DedispersionTuning.cpp:    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
src/DedispersionTuning.cpp:    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
src/DedispersionTuning.cpp:    std::cerr << argv[0] << " -iterations ... -opencl_platform ... -opencl_device ... [-best] [-single_step | -step_one | -step_two] -padding ... -vector ... -min_threads ... -max_threads ... -max_columns ... -max_rows ... -max_items ... -max_sample_items ... -max_dm_items ... -max_unroll ... -beams ... -samples ... -sampling_time ... -min_freq ... -channel_bandwidth ... -channels ... " << std::endl;
src/DedispersionTuning.cpp:  isa::OpenCL::OpenCLRunTime openCLRunTime;
src/DedispersionTuning.cpp:      isa::OpenCL::initializeOpenCL(clPlatformID, 1, openCLRunTime);
src/DedispersionTuning.cpp:          initializeDeviceMemorySingleStep(*(openCLRunTime.context), &(openCLRunTime.queues->at(clDeviceID)[0]), shiftsSingleStep, &shiftsSingleStep_d, zappedChannels, &zappedChannels_d, beamMappingSingleStep, &beamMappingSingleStep_d, dispersedData_size, &dispersedData_d, dedispersedData_size, &dedispersedData_d);
src/DedispersionTuning.cpp:          initializeDeviceMemoryStepOne(*(openCLRunTime.context), &(openCLRunTime.queues->at(clDeviceID)[0]), shiftsStepOne, &shiftsStepOne_d, zappedChannels, &zappedChannels_d, dispersedData_size, &dispersedData_d, subbandedData_size, &subbandedData_d);
src/DedispersionTuning.cpp:          initializeDeviceMemoryStepTwo(*(openCLRunTime.context), &(openCLRunTime.queues->at(clDeviceID)[0]), shiftsStepTwo, &shiftsStepTwo_d, beamMappingStepTwo, &beamMappingStepTwo_d, subbandedData_size, &subbandedData_d, dedispersedData_size, &dedispersedData_d);
src/DedispersionTuning.cpp:      code = Dedispersion::getDedispersionOpenCL< inputDataType, outputDataType >(*conf, padding, inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsSingleStep);
src/DedispersionTuning.cpp:      code = Dedispersion::getSubbandDedispersionStepOneOpenCL< inputDataType, outputDataType >(*conf, padding, inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
src/DedispersionTuning.cpp:      code = Dedispersion::getSubbandDedispersionStepTwoOpenCL< outputDataType >(*conf, padding, outputDataName, observation, *shiftsStepTwo);
src/DedispersionTuning.cpp:        kernel = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTuning.cpp:        kernel = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTuning.cpp:        kernel = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTuning.cpp:    } catch ( isa::OpenCL::OpenCLError & err ) {
src/DedispersionTuning.cpp:      openCLRunTime.queues->at(clDeviceID)[0].finish();
src/DedispersionTuning.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
src/DedispersionTuning.cpp:        openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local, 0, &event);
src/DedispersionTuning.cpp:      std::cerr << "OpenCL error kernel execution (";
src/DedispersionTuning.cpp:    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
src/DedispersionTuning.cpp:    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
src/DedispersionTuning.cpp:    std::cerr << "OpenCL error: " << std::to_string(err.err()) << "." << std::endl;
src/DedispersionTest.cpp:#include <InitializeOpenCL.hpp>
src/DedispersionTest.cpp:    clPlatformID = args.getSwitchArgument< unsigned int >("-opencl_platform");
src/DedispersionTest.cpp:    clDeviceID = args.getSwitchArgument< unsigned int >("-opencl_device");
src/DedispersionTest.cpp:    std::cerr << "Usage: " << argv[0] << " [-print_code] [-print_results] [-random] [-single_step | -step_one | -step_two] -opencl_platform ... -opencl_device ... -padding ... [-local] -threadsD0 ... -threadsD1 ... -itemsD0 ... -itemsD1 ... -unroll ... -beams ... -channels ... -min_freq ... -channel_bandwidth ... -samples ... -sampling_time ..." << std::endl;
src/DedispersionTest.cpp:  // Initialize OpenCL
src/DedispersionTest.cpp:  isa::OpenCL::OpenCLRunTime openCLRunTime;
src/DedispersionTest.cpp:  isa::OpenCL::initializeOpenCL(clPlatformID, 1, openCLRunTime);
src/DedispersionTest.cpp:      shiftsSingleStep_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, shiftsSingleStep->size() * sizeof(float), 0, 0);
src/DedispersionTest.cpp:      zappedChannels_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(unsigned int), 0, 0);
src/DedispersionTest.cpp:      dispersedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, dispersedData.size() * sizeof(inputDataType), 0, 0);
src/DedispersionTest.cpp:      dedispersedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_WRITE_ONLY, dedispersedData.size() * sizeof(outputDataType), 0, 0);
src/DedispersionTest.cpp:      beamMappingSingleStep_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, beamMappingSingleStep.size() * sizeof(unsigned int), 0, 0);
src/DedispersionTest.cpp:      shiftsStepOne_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, shiftsStepOne->size() * sizeof(float), 0, 0);
src/DedispersionTest.cpp:      zappedChannels_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, zappedChannels.size() * sizeof(unsigned int), 0, 0);
src/DedispersionTest.cpp:      dispersedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, dispersedData.size() * sizeof(inputDataType), 0, 0);
src/DedispersionTest.cpp:      subbandedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_WRITE_ONLY, subbandedData.size() * sizeof(outputDataType), 0, 0);
src/DedispersionTest.cpp:      shiftsStepTwo_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, shiftsStepTwo->size() * sizeof(float), 0, 0);
src/DedispersionTest.cpp:      subbandedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, subbandedData.size() * sizeof(outputDataType), 0, 0);
src/DedispersionTest.cpp:      dedispersedData_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_WRITE_ONLY, dedispersedData.size() * sizeof(outputDataType), 0, 0);
src/DedispersionTest.cpp:      beamMappingStepTwo_d = cl::Buffer(*(openCLRunTime.context), CL_MEM_READ_ONLY, beamMappingStepTwo.size() * sizeof(unsigned int), 0, 0);
src/DedispersionTest.cpp:    std::cerr << "OpenCL error allocating memory: " << std::to_string(err.err()) << "." << std::endl;
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(shiftsSingleStep_d, CL_FALSE, 0, shiftsSingleStep->size() * sizeof(float), reinterpret_cast< void * >(shiftsSingleStep->data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(zappedChannels_d, CL_FALSE, 0, zappedChannels.size() * sizeof(unsigned int), reinterpret_cast< void * >(zappedChannels.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(beamMappingSingleStep_d, CL_FALSE, 0, beamMappingSingleStep.size() * sizeof(unsigned int), reinterpret_cast< void * >(beamMappingSingleStep.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(shiftsStepOne_d, CL_FALSE, 0, shiftsStepOne->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepOne->data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(zappedChannels_d, CL_FALSE, 0, zappedChannels.size() * sizeof(unsigned int), reinterpret_cast< void * >(zappedChannels.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(dispersedData_d, CL_FALSE, 0, dispersedData.size() * sizeof(inputDataType), reinterpret_cast< void * >(dispersedData.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(shiftsStepTwo_d, CL_FALSE, 0, shiftsStepTwo->size() * sizeof(float), reinterpret_cast< void * >(shiftsStepTwo->data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(subbandedData_d, CL_FALSE, 0, subbandedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(subbandedData.data()), 0, 0);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueWriteBuffer(beamMappingStepTwo_d, CL_FALSE, 0, beamMappingStepTwo.size() * sizeof(unsigned int), reinterpret_cast< void * >(beamMappingStepTwo.data()), 0, 0);
src/DedispersionTest.cpp:    std::cerr << "OpenCL error H2D transfer: " << std::to_string(err.err()) << "." << std::endl;
src/DedispersionTest.cpp:    code = Dedispersion::getDedispersionOpenCL< inputDataType, outputDataType >(conf, padding, inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsSingleStep);
src/DedispersionTest.cpp:    code = Dedispersion::getSubbandDedispersionStepOneOpenCL< inputDataType, outputDataType >(conf, padding, inputBits, inputDataName, intermediateDataName, outputDataName, observation, *shiftsStepOne);
src/DedispersionTest.cpp:    code = Dedispersion::getSubbandDedispersionStepTwoOpenCL< outputDataType >(conf, padding, outputDataName, observation, *shiftsStepTwo);
src/DedispersionTest.cpp:      kernel = isa::OpenCL::compile("dedispersion", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTest.cpp:      kernel = isa::OpenCL::compile("dedispersionStepOne", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTest.cpp:      kernel = isa::OpenCL::compile("dedispersionStepTwo", *code, "-cl-mad-enable -Werror", *(openCLRunTime.context), openCLRunTime.devices->at(clDeviceID));
src/DedispersionTest.cpp:  } catch ( isa::OpenCL::OpenCLError & err ) {
src/DedispersionTest.cpp:  // Run OpenCL kernel and CPU control
src/DedispersionTest.cpp:    openCLRunTime.queues->at(clDeviceID)[0].enqueueNDRangeKernel(*kernel, cl::NullRange, global, local);
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()));
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(subbandedData_d, CL_TRUE, 0, subbandedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(subbandedData.data()));
src/DedispersionTest.cpp:      openCLRunTime.queues->at(clDeviceID)[0].enqueueReadBuffer(dedispersedData_d, CL_TRUE, 0, dedispersedData.size() * sizeof(outputDataType), reinterpret_cast< void * >(dedispersedData.data()));
src/DedispersionTest.cpp:    std::cerr << "OpenCL error kernel execution: " << std::to_string(err.err()) << "." << std::endl;

```
