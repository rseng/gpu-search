# https://github.com/lbcb-sci/racon

```console
test/racon_test.cpp:        int8_t match, int8_t mismatch, int8_t gap, uint32_t cuda_batches = 0,
test/racon_test.cpp:        bool cuda_banded_alignment = false, uint32_t cudaaligner_batches = 0) {
test/racon_test.cpp:            mismatch, gap, 4, cuda_batches, cuda_banded_alignment, cudaaligner_batches);
test/racon_test.cpp:#ifdef CUDA_ENABLED
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithQualitiesCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithoutQualitiesCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithQualitiesAndAlignmentsCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithoutQualitiesAndWithAlignmentsCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithQualitiesLargerWindowCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, ConsensusWithQualitiesEditDistanceCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, FragmentCorrectionWithQualitiesCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, FragmentCorrectionWithQualitiesFullCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, FragmentCorrectionWithoutQualitiesFullCUDA) {
test/racon_test.cpp:TEST_F(RaconPolishingTest, FragmentCorrectionWithQualitiesFullMhapCUDA) {
README.md:### CUDA Support
README.md:3. CUDA 9.0+
README.md:### CUDA Support
README.md:Racon makes use of [NVIDIA's GenomeWorks SDK](https://github.com/clara-parabricks/GenomeWorks) for CUDA accelerated polishing and alignment.
README.md:To build `racon` with CUDA support, add `-Dracon_enable_cuda=ON` while running `cmake`. If CUDA support is unavailable, the `cmake` step will error out.
README.md:Note that the CUDA support flag does not produce a new binary target. Instead it augments the existing `racon` binary itself.
README.md:cmake -DCMAKE_BUILD_TYPE=Release -Dracon_enable_cuda=ON ..
README.md:***Note***: Short read polishing with CUDA is still in development!
README.md:    only available when built with CUDA:
README.md:        -c, --cudapoa-batches <int>
README.md:            number of batches for CUDA accelerated polishing per GPU
README.md:        -b, --cuda-banded-alignment
README.md:            use banding approximation for polishing on GPU. Only applicable when -c is used.
README.md:        --cudaaligner-batches <int>
README.md:            number of batches for CUDA accelerated alignment per GPU
README.md:        --cudaaligner-band-width <int>
README.md:            Band width for cuda alignment. Must be >= 0. Non-zero allows user defined
CMakeLists.txt:option(racon_enable_cuda "Build with NVIDIA CUDA support" OFF)
CMakeLists.txt:if (racon_enable_cuda)
CMakeLists.txt:  find_package(CUDA 9.0 QUIET REQUIRED)
CMakeLists.txt:  if (NOT ${CUDA_FOUND})
CMakeLists.txt:    message(FATAL_ERROR "CUDA not detected on system. Please install")
CMakeLists.txt:    message(STATUS "Using CUDA ${CUDA_VERSION} from ${CUDA_TOOLKIT_ROOT_DIR}")
CMakeLists.txt:    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
CMakeLists.txt:    find_package(cudapoa REQUIRED)
CMakeLists.txt:    find_package(cudaaligner REQUIRED)
CMakeLists.txt:    if (NOT TARGET cudapoa)
CMakeLists.txt:    if (NOT TARGET cudaaligner)
CMakeLists.txt:if (racon_enable_cuda)
CMakeLists.txt:    src/cuda/cudapolisher.cpp
CMakeLists.txt:    src/cuda/cudabatch.cpp
CMakeLists.txt:    src/cuda/cudaaligner.cpp)
CMakeLists.txt:  cuda_add_library(racon
CMakeLists.txt:    PRIVATE CUDA_ENABLED)
CMakeLists.txt:if (racon_enable_cuda)
CMakeLists.txt:    cudapoa
CMakeLists.txt:    cudaaligner)
CMakeLists.txt:if (racon_enable_cuda)
CMakeLists.txt:    PRIVATE CUDA_ENABLED)
CMakeLists.txt:  if (racon_enable_cuda)
CMakeLists.txt:      PRIVATE CUDA_ENABLED)
CMakeLists.txt:  if (racon_enable_cuda)
CMakeLists.txt:    set(racon_wrapper_enable_cuda True)
CMakeLists.txt:    set(racon_wrapper_enable_cuda False)
scripts/racon_wrapper.py:        error_threshold, match, mismatch, gap, threads, cudaaligner_band_width,
scripts/racon_wrapper.py:        cudaaligner_batches, cudapoa_batches, cuda_banded_alignment):
scripts/racon_wrapper.py:        self.cudaaligner_band_width = cudaaligner_band_width
scripts/racon_wrapper.py:        self.cudaaligner_batches = cudaaligner_batches
scripts/racon_wrapper.py:        self.cudapoa_batches = cudapoa_batches
scripts/racon_wrapper.py:        self.cuda_banded_alignment = cuda_banded_alignment
scripts/racon_wrapper.py:        if (@racon_wrapper_enable_cuda@):
scripts/racon_wrapper.py:            if (self.cuda_banded_alignment == True): racon_params.append('-b')
scripts/racon_wrapper.py:                '--cudaaligner-band-width', str(self.cudaaligner_band_width),
scripts/racon_wrapper.py:                '--cudaaligner-batches', str(self.cudaaligner_batches),
scripts/racon_wrapper.py:                '-c', str(self.cudapoa_batches)])
scripts/racon_wrapper.py:    parser.add_argument('--cudaaligner-band-width', default=0, help='''Band
scripts/racon_wrapper.py:        width for cuda alignment. Must be >= 0. Non-zero allows user defined
scripts/racon_wrapper.py:    parser.add_argument('--cudaaligner-batches', default=0, help='''number of
scripts/racon_wrapper.py:        batches for CUDA accelerated alignment''')
scripts/racon_wrapper.py:    parser.add_argument('-c', '--cudapoa-batches', default=0, help='''number of
scripts/racon_wrapper.py:        batches for CUDA accelerated polishing''')
scripts/racon_wrapper.py:    parser.add_argument('-b', '--cuda-banded-alignment', action='store_true',
scripts/racon_wrapper.py:        help='''use banding approximation for polishing on GPU. Only applicable
scripts/racon_wrapper.py:        args.cudaaligner_band_width, args.cudaaligner_batches,
scripts/racon_wrapper.py:        args.cudapoa_batches, args.cuda_banded_alignment)
src/overlap.hpp:#ifdef CUDA_ENABLED
src/overlap.hpp:    friend class CUDABatchAligner;
src/window.hpp:#ifdef CUDA_ENABLED
src/window.hpp:    friend class CUDABatchProcessor;
src/main.cpp:#ifdef CUDA_ENABLED
src/main.cpp:#include "cuda/cudapolisher.hpp"
src/main.cpp:static const int32_t CUDAALIGNER_INPUT_CODE = 10000;
src/main.cpp:static const int32_t CUDAALIGNER_BAND_WIDTH_INPUT_CODE = 10001;
src/main.cpp:#ifdef CUDA_ENABLED
src/main.cpp:    {"cudapoa-batches", optional_argument, 0, 'c'},
src/main.cpp:    {"cuda-banded-alignment", no_argument, 0, 'b'},
src/main.cpp:    {"cudaaligner-batches", required_argument, 0, CUDAALIGNER_INPUT_CODE},
src/main.cpp:    {"cudaaligner-band-width", required_argument, 0, CUDAALIGNER_BAND_WIDTH_INPUT_CODE},
src/main.cpp:    uint32_t cudapoa_batches = 0;
src/main.cpp:    uint32_t cudaaligner_batches = 0;
src/main.cpp:    uint32_t cudaaligner_band_width = 0;
src/main.cpp:    bool cuda_banded_alignment = false;
src/main.cpp:#ifdef CUDA_ENABLED
src/main.cpp:#ifdef CUDA_ENABLED
src/main.cpp:                //if option c encountered, cudapoa_batches initialized with a default value of 1.
src/main.cpp:                cudapoa_batches = 1;
src/main.cpp:                    cudapoa_batches = atoi(argv[optind++]);
src/main.cpp:                    cudapoa_batches = atoi(optarg);
src/main.cpp:                cuda_banded_alignment = true;
src/main.cpp:            case CUDAALIGNER_INPUT_CODE: // cudaaligner-batches
src/main.cpp:                cudaaligner_batches = atoi(optarg);
src/main.cpp:            case CUDAALIGNER_BAND_WIDTH_INPUT_CODE: // cudaaligner-band-width
src/main.cpp:                cudaaligner_band_width = atoi(optarg);
src/main.cpp:        cudapoa_batches, cuda_banded_alignment, cudaaligner_batches,
src/main.cpp:        cudaaligner_band_width);
src/main.cpp:#ifdef CUDA_ENABLED
src/main.cpp:        "        -c, --cudapoa-batches <int>\n"
src/main.cpp:        "            number of batches for CUDA accelerated polishing per GPU\n"
src/main.cpp:        "        -b, --cuda-banded-alignment\n"
src/main.cpp:        "            use banding approximation for alignment on GPU\n"
src/main.cpp:        "        --cudaaligner-batches <int>\n"
src/main.cpp:        "            number of batches for CUDA accelerated alignment per GPU\n"
src/main.cpp:        "        --cudaaligner-band-width <int>\n"
src/main.cpp:        "            Band width for cuda alignment. Must be >= 0. Non-zero allows user defined \n"
src/polisher.cpp:#ifdef CUDA_ENABLED
src/polisher.cpp:#include "cuda/cudapolisher.hpp"
src/polisher.cpp:    uint32_t num_threads, uint32_t cudapoa_batches, bool cuda_banded_alignment,
src/polisher.cpp:    uint32_t cudaaligner_batches, uint32_t cudaaligner_band_width) {
src/polisher.cpp:    if (cudapoa_batches > 0 || cudaaligner_batches > 0)
src/polisher.cpp:#ifdef CUDA_ENABLED
src/polisher.cpp:        // If CUDA is enabled, return an instance of the CUDAPolisher object.
src/polisher.cpp:        return std::unique_ptr<Polisher>(new CUDAPolisher(std::move(sparser),
src/polisher.cpp:                    num_threads, cudapoa_batches, cuda_banded_alignment, cudaaligner_batches,
src/polisher.cpp:                    cudaaligner_band_width));
src/polisher.cpp:                "Attemping to use CUDA when CUDA support is not available.\n"
src/polisher.cpp:        (void) cuda_banded_alignment;
src/polisher.cpp:        (void) cudaaligner_band_width;
src/polisher.hpp:    uint32_t num_threads, uint32_t cuda_batches = 0,
src/polisher.hpp:    bool cuda_banded_alignment = false, uint32_t cudaaligner_batches = 0,
src/polisher.hpp:    uint32_t cudaaligner_band_width = 0);
src/polisher.hpp:        uint32_t num_threads, uint32_t cuda_batches, bool cuda_banded_alignment,
src/polisher.hpp:        uint32_t cudaaligner_batches, uint32_t cudaaligner_band_width);
src/cuda/cudaaligner.hpp:* @file cudaaligner.hpp
src/cuda/cudaaligner.hpp: * @brief CUDA aligner class header file
src/cuda/cudaaligner.hpp:#include <claraparabricks/genomeworks/cudaaligner/cudaaligner.hpp>
src/cuda/cudaaligner.hpp:#include <claraparabricks/genomeworks/cudaaligner/aligner.hpp>
src/cuda/cudaaligner.hpp:#include <claraparabricks/genomeworks/cudaaligner/alignment.hpp>
src/cuda/cudaaligner.hpp:class CUDABatchAligner;
src/cuda/cudaaligner.hpp:std::unique_ptr<CUDABatchAligner> createCUDABatchAligner(uint32_t max_bandwidth, uint32_t device_id, int64_t max_gpu_memory);
src/cuda/cudaaligner.hpp:class CUDABatchAligner
src/cuda/cudaaligner.hpp:        virtual ~CUDABatchAligner();
src/cuda/cudaaligner.hpp:         * @brief Runs batched alignment of overlaps on GPU.
src/cuda/cudaaligner.hpp:         *        copmuted on the GPU.
src/cuda/cudaaligner.hpp:        // Builder function to create a new CUDABatchAligner object.
src/cuda/cudaaligner.hpp:        friend std::unique_ptr<CUDABatchAligner>
src/cuda/cudaaligner.hpp:        createCUDABatchAligner(uint32_t max_bandwidth, uint32_t device_id, int64_t max_gpu_memory);
src/cuda/cudaaligner.hpp:        CUDABatchAligner(uint32_t max_bandwidth, uint32_t device_id, int64_t max_gpu_memory);
src/cuda/cudaaligner.hpp:        CUDABatchAligner(const CUDABatchAligner&) = delete;
src/cuda/cudaaligner.hpp:        const CUDABatchAligner& operator=(const CUDABatchAligner&) = delete;
src/cuda/cudaaligner.hpp:        std::unique_ptr<claraparabricks::genomeworks::cudaaligner::Aligner> aligner_;
src/cuda/cudaaligner.hpp:        // CUDA stream for batch.
src/cuda/cudaaligner.hpp:        cudaStream_t stream_;
src/cuda/cudabatch.hpp:* @file cudabatch.hpp
src/cuda/cudabatch.hpp: * @brief CUDA batch class header file
src/cuda/cudabatch.hpp:#include <cuda_runtime_api.h>
src/cuda/cudabatch.hpp:#include <claraparabricks/genomeworks/cudapoa/batch.hpp>
src/cuda/cudabatch.hpp:class CUDABatchProcessor;
src/cuda/cudabatch.hpp:std::unique_ptr<CUDABatchProcessor> createCUDABatch(uint32_t max_window_depth, uint32_t device, size_t avail_mem, int8_t gap, int8_t mismatch, int8_t match, bool cuda_banded_alignment);
src/cuda/cudabatch.hpp:class CUDABatchProcessor
src/cuda/cudabatch.hpp:    ~CUDABatchProcessor();
src/cuda/cudabatch.hpp:    // Builder function to create a new CUDABatchProcessor object.
src/cuda/cudabatch.hpp:    friend std::unique_ptr<CUDABatchProcessor>
src/cuda/cudabatch.hpp:    createCUDABatch(uint32_t max_window_depth, uint32_t device, size_t avail_mem, int8_t gap, int8_t mismatch, int8_t match, bool cuda_banded_alignment);
src/cuda/cudabatch.hpp:     * @brief Constructor for CUDABatch class.
src/cuda/cudabatch.hpp:     * @param[in] cuda_banded_alignment : Use banded POA alignment
src/cuda/cudabatch.hpp:    CUDABatchProcessor(uint32_t max_window_depth, uint32_t device, size_t avail_mem, int8_t gap, int8_t mismatch, int8_t match, bool cuda_banded_alignment);
src/cuda/cudabatch.hpp:    CUDABatchProcessor(const CUDABatchProcessor&) = delete;
src/cuda/cudabatch.hpp:    const CUDABatchProcessor& operator=(const CUDABatchProcessor&) = delete;
src/cuda/cudabatch.hpp:     * @brief Run the CUDA kernel for generating POA on the batch.
src/cuda/cudabatch.hpp:    // CUDA-POA library object that manages POA batch.
src/cuda/cudabatch.hpp:    std::unique_ptr<claraparabricks::genomeworks::cudapoa::Batch> cudapoa_batch_;
src/cuda/cudabatch.hpp:    cudaStream_t stream_;
src/cuda/cudabatch.cpp: * @file cudabatch.cpp
src/cuda/cudabatch.cpp: * @brief CUDABatch class source file
src/cuda/cudabatch.cpp:#include "cudabatch.hpp"
src/cuda/cudabatch.cpp:#include "cudautils.hpp"
src/cuda/cudabatch.cpp:#include <claraparabricks/genomeworks/utils/cudautils.hpp>
src/cuda/cudabatch.cpp:using namespace claraparabricks::genomeworks::cudapoa;
src/cuda/cudabatch.cpp:std::atomic<uint32_t> CUDABatchProcessor::batches;
src/cuda/cudabatch.cpp:std::unique_ptr<CUDABatchProcessor> createCUDABatch(uint32_t max_window_depth,
src/cuda/cudabatch.cpp:                                                    bool cuda_banded_alignment)
src/cuda/cudabatch.cpp:    return std::unique_ptr<CUDABatchProcessor>(new CUDABatchProcessor(max_window_depth,
src/cuda/cudabatch.cpp:                                                                      cuda_banded_alignment));
src/cuda/cudabatch.cpp:CUDABatchProcessor::CUDABatchProcessor(uint32_t max_window_depth,
src/cuda/cudabatch.cpp:                                       bool cuda_banded_alignment)
src/cuda/cudabatch.cpp:    bid_ = CUDABatchProcessor::batches++;
src/cuda/cudabatch.cpp:    // Create new CUDA stream.
src/cuda/cudabatch.cpp:    GW_CU_CHECK_ERR(cudaStreamCreate(&stream_));
src/cuda/cudabatch.cpp:                             cuda_banded_alignment ? BandMode::static_band : BandMode::full_band);
src/cuda/cudabatch.cpp:    cudapoa_batch_ = create_batch(device,
src/cuda/cudabatch.cpp:CUDABatchProcessor::~CUDABatchProcessor()
src/cuda/cudabatch.cpp:    // Destroy CUDA stream.
src/cuda/cudabatch.cpp:    GW_CU_CHECK_ERR(cudaStreamDestroy(stream_));
src/cuda/cudabatch.cpp:bool CUDABatchProcessor::addWindow(std::shared_ptr<Window> window)
src/cuda/cudabatch.cpp:    // Add group to CUDAPOA batch object.
src/cuda/cudabatch.cpp:    StatusType status = cudapoa_batch_->add_poa_group(entry_status, poa_group);
src/cuda/cudabatch.cpp:                    cudapoa_batch_->batch_id());
src/cuda/cudabatch.cpp:bool CUDABatchProcessor::hasWindows() const
src/cuda/cudabatch.cpp:    return (cudapoa_batch_->get_total_poas() > 0);
src/cuda/cudabatch.cpp:void CUDABatchProcessor::convertPhredQualityToWeights(const char* qual,
src/cuda/cudabatch.cpp:void CUDABatchProcessor::generatePOA()
src/cuda/cudabatch.cpp:    cudapoa_batch_->generate_poa();
src/cuda/cudabatch.cpp:void CUDABatchProcessor::getConsensus()
src/cuda/cudabatch.cpp:    cudapoa_batch_->get_consensus(consensuses, coverages, output_status);
src/cuda/cudabatch.cpp:            // TODO: We still run this case through the GPU, but could take it out.
src/cuda/cudabatch.cpp:                        fprintf(stderr, "[CUDABatchProcessor] warning: "
src/cuda/cudabatch.cpp:const std::vector<bool>& CUDABatchProcessor::generateConsensus()
src/cuda/cudabatch.cpp:void CUDABatchProcessor::reset()
src/cuda/cudabatch.cpp:    cudapoa_batch_->reset();
src/cuda/cudapolisher.cpp: * @file cudapolisher.cpp
src/cuda/cudapolisher.cpp: * @brief CUDA Polisher class source file
src/cuda/cudapolisher.cpp:#include <cuda_profiler_api.h>
src/cuda/cudapolisher.cpp:#include "cudapolisher.hpp"
src/cuda/cudapolisher.cpp:#include <claraparabricks/genomeworks/utils/cudautils.hpp>
src/cuda/cudapolisher.cpp:CUDAPolisher::CUDAPolisher(std::unique_ptr<bioparser::Parser<Sequence>> sparser,
src/cuda/cudapolisher.cpp:    uint32_t num_threads, uint32_t cudapoa_batches, bool cuda_banded_alignment,
src/cuda/cudapolisher.cpp:    uint32_t cudaaligner_batches, uint32_t cudaaligner_band_width)
src/cuda/cudapolisher.cpp:        , cudapoa_batches_(cudapoa_batches)
src/cuda/cudapolisher.cpp:        , cudaaligner_batches_(cudaaligner_batches)
src/cuda/cudapolisher.cpp:        , cuda_banded_alignment_(cuda_banded_alignment)
src/cuda/cudapolisher.cpp:        , cudaaligner_band_width_(cudaaligner_band_width)
src/cuda/cudapolisher.cpp:    claraparabricks::genomeworks::cudapoa::Init();
src/cuda/cudapolisher.cpp:    claraparabricks::genomeworks::cudaaligner::Init();
src/cuda/cudapolisher.cpp:    GW_CU_CHECK_ERR(cudaGetDeviceCount(&num_devices_));
src/cuda/cudapolisher.cpp:        throw std::runtime_error("No GPU devices found.");
src/cuda/cudapolisher.cpp:    std::cerr << "Using " << num_devices_ << " GPU(s) to perform polishing" << std::endl;
src/cuda/cudapolisher.cpp:    // Run dummy call on each device to initialize CUDA context.
src/cuda/cudapolisher.cpp:        GW_CU_CHECK_ERR(cudaSetDevice(dev_id));
src/cuda/cudapolisher.cpp:        GW_CU_CHECK_ERR(cudaFree(0));
src/cuda/cudapolisher.cpp:    std::cerr << "[CUDAPolisher] Constructed." << std::endl;
src/cuda/cudapolisher.cpp:CUDAPolisher::~CUDAPolisher()
src/cuda/cudapolisher.cpp:    cudaDeviceSynchronize();
src/cuda/cudapolisher.cpp:    cudaProfilerStop();
src/cuda/cudapolisher.cpp:void CUDAPolisher::find_overlap_breaking_points(std::vector<std::unique_ptr<Overlap>>& overlaps)
src/cuda/cudapolisher.cpp:    if (cudaaligner_batches_ >= 1)
src/cuda/cudapolisher.cpp:        auto fill_next_batch = [&mutex_overlaps, &next_overlap_index, &overlaps, this](CUDABatchAligner* batch) -> std::pair<uint32_t, uint32_t> {
src/cuda/cudapolisher.cpp:        auto process_batch = [&fill_next_batch, &logger_step, &log_bar_idx, &log_bar_idx_prev, &window_idx, &mutex_log_bar_idx, this](CUDABatchAligner* batch) -> void {
src/cuda/cudapolisher.cpp:                            logger_->bar("[racon::CUDAPolisher::initialize] aligning overlaps");
src/cuda/cudapolisher.cpp:        // and use that to calculate cudaaligner batch size.
src/cuda/cudapolisher.cpp:        if (cudaaligner_band_width_ == 0)
src/cuda/cudapolisher.cpp:            cudaaligner_band_width_ = static_cast<uint32_t>(mean * 0.1f);
src/cuda/cudapolisher.cpp:            GW_CU_CHECK_ERR(cudaSetDevice(device));
src/cuda/cudapolisher.cpp:            GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
src/cuda/cudapolisher.cpp:            const int64_t usable_memory_per_aligner = free_usable_memory / cudaaligner_batches_;
src/cuda/cudapolisher.cpp:            const int32_t max_bandwidth = cudaaligner_band_width_ & ~0x1; // Band width needs to be even
src/cuda/cudapolisher.cpp:            std::cerr << "GPU " << device << ": Aligning with band width " << max_bandwidth << std::endl;
src/cuda/cudapolisher.cpp:            for(uint32_t batch = 0; batch < cudaaligner_batches_; batch++)
src/cuda/cudapolisher.cpp:                batch_aligners_.emplace_back(createCUDABatchAligner(max_bandwidth, device, usable_memory_per_aligner));
src/cuda/cudapolisher.cpp:        logger_->log("[racon::CUDAPolisher::initialize] allocated memory on GPUs for alignment");
src/cuda/cudapolisher.cpp:        // Determine overlaps missed by GPU which will fall back to CPU.
src/cuda/cudapolisher.cpp:        std::cerr << "Alignment skipped by GPU: " << missing_overlaps << " / " << overlaps.size() << std::endl;
src/cuda/cudapolisher.cpp:    // Any overlaps that couldn't be processed by the GPU are also handled here
src/cuda/cudapolisher.cpp:void CUDAPolisher::polish(std::vector<std::unique_ptr<Sequence>>& dst,
src/cuda/cudapolisher.cpp:    if (cudapoa_batches_ < 1)
src/cuda/cudapolisher.cpp:            GW_CU_CHECK_ERR(cudaSetDevice(device));
src/cuda/cudapolisher.cpp:            GW_CU_CHECK_ERR(cudaMemGetInfo(&free, &total));
src/cuda/cudapolisher.cpp:            size_t mem_per_batch = 0.9 * free / cudapoa_batches_;
src/cuda/cudapolisher.cpp:            for(uint32_t batch = 0; batch < cudapoa_batches_; batch++)
src/cuda/cudapolisher.cpp:                batch_processors_.emplace_back(createCUDABatch(MAX_DEPTH_PER_WINDOW, device, mem_per_batch, gap_, mismatch_, match_, cuda_banded_alignment_));
src/cuda/cudapolisher.cpp:        logger_->log("[racon::CUDAPolisher::polish] allocated memory on GPUs for polishing");
src/cuda/cudapolisher.cpp:        auto fill_next_batch = [&mutex_windows, &next_window_index, this](CUDABatchProcessor* batch) -> std::pair<uint32_t, uint32_t> {
src/cuda/cudapolisher.cpp:        auto process_batch = [&fill_next_batch, &logger_step, &log_bar_idx, &mutex_log_bar_idx, &window_idx, &log_bar_idx_prev, this](CUDABatchProcessor* batch) -> void {
src/cuda/cudapolisher.cpp:                    // result vector of the CUDAPolisher.
src/cuda/cudapolisher.cpp:                                logger_->bar("[racon::CUDAPolisher::polish] generating consensus");
src/cuda/cudapolisher.cpp:        logger_->log("[racon::CUDAPolisher::polish] polished windows on GPU");
src/cuda/cudapolisher.cpp:        // Start timing CPU time for failed windows on GPU
src/cuda/cudapolisher.cpp:            logger_->log("[racon::CUDAPolisher::polish] polished remaining windows on CPU");
src/cuda/cudapolisher.cpp:        logger_->log("[racon::CUDAPolisher::polish] generated consensus");
src/cuda/cudautils.hpp:// Implementation file for CUDA POA utilities.
src/cuda/cudautils.hpp:#include <cuda_runtime_api.h>
src/cuda/cudautils.hpp:void cudaCheckError(std::string &msg)
src/cuda/cudautils.hpp:    cudaError_t error = cudaGetLastError();
src/cuda/cudautils.hpp:    if (error != cudaSuccess)
src/cuda/cudautils.hpp:        fprintf(stderr, "%s (CUDA error %s)\n", msg.c_str(), cudaGetErrorString(error));
src/cuda/cudapolisher.hpp: * @file cudapolisher.hpp
src/cuda/cudapolisher.hpp: * @brief CUDA Polisher class header file
src/cuda/cudapolisher.hpp:#include "cudabatch.hpp"
src/cuda/cudapolisher.hpp:#include "cudaaligner.hpp"
src/cuda/cudapolisher.hpp:class CUDAPolisher : public Polisher {
src/cuda/cudapolisher.hpp:    ~CUDAPolisher();
src/cuda/cudapolisher.hpp:        uint32_t num_threads, uint32_t cudapoa_batches, bool cuda_banded_alignment,
src/cuda/cudapolisher.hpp:        uint32_t cudaaligner_batches, uint32_t cudaaligner_band_width);
src/cuda/cudapolisher.hpp:    CUDAPolisher(std::unique_ptr<bioparser::Parser<Sequence>> sparser,
src/cuda/cudapolisher.hpp:        uint32_t num_threads, uint32_t cudapoa_batches, bool cuda_banded_alignment,
src/cuda/cudapolisher.hpp:        uint32_t cudaaligner_batches, uint32_t cudaaligner_band_width);
src/cuda/cudapolisher.hpp:    CUDAPolisher(const CUDAPolisher&) = delete;
src/cuda/cudapolisher.hpp:    const CUDAPolisher& operator=(const CUDAPolisher&) = delete;
src/cuda/cudapolisher.hpp:    static std::vector<uint32_t> calculate_batches_per_gpu(uint32_t cudapoa_batches, uint32_t gpus);
src/cuda/cudapolisher.hpp:    std::vector<std::unique_ptr<CUDABatchProcessor>> batch_processors_;
src/cuda/cudapolisher.hpp:    std::vector<std::unique_ptr<CUDABatchAligner>> batch_aligners_;
src/cuda/cudapolisher.hpp:    uint32_t cudapoa_batches_;
src/cuda/cudapolisher.hpp:    uint32_t cudaaligner_batches_;
src/cuda/cudapolisher.hpp:    // Number of GPU devices to run with.
src/cuda/cudapolisher.hpp:    bool cuda_banded_alignment_;
src/cuda/cudapolisher.hpp:    uint32_t cudaaligner_band_width_;
src/cuda/cudaaligner.cpp: * @file cudaaligner.cpp
src/cuda/cudaaligner.cpp: * @brief CUDABatchAligner class source file
src/cuda/cudaaligner.cpp:#include <claraparabricks/genomeworks/utils/cudautils.hpp>
src/cuda/cudaaligner.cpp:#include "cudaaligner.hpp"
src/cuda/cudaaligner.cpp:using namespace claraparabricks::genomeworks::cudaaligner;
src/cuda/cudaaligner.cpp:std::atomic<uint32_t> CUDABatchAligner::batches;
src/cuda/cudaaligner.cpp:std::unique_ptr<CUDABatchAligner> createCUDABatchAligner(uint32_t max_bandwidth,
src/cuda/cudaaligner.cpp:                                                         int64_t max_gpu_memory)
src/cuda/cudaaligner.cpp:    return std::unique_ptr<CUDABatchAligner>(new CUDABatchAligner(max_bandwidth,
src/cuda/cudaaligner.cpp:                                                                  max_gpu_memory));
src/cuda/cudaaligner.cpp:CUDABatchAligner::CUDABatchAligner(uint32_t max_bandwidth,
src/cuda/cudaaligner.cpp:                                   int64_t max_gpu_memory)
src/cuda/cudaaligner.cpp:    bid_ = CUDABatchAligner::batches++;
src/cuda/cudaaligner.cpp:    GW_CU_CHECK_ERR(cudaSetDevice(device_id));
src/cuda/cudaaligner.cpp:    GW_CU_CHECK_ERR(cudaStreamCreate(&stream_));
src/cuda/cudaaligner.cpp:                              max_gpu_memory);
src/cuda/cudaaligner.cpp:CUDABatchAligner::~CUDABatchAligner()
src/cuda/cudaaligner.cpp:    GW_CU_CHECK_ERR(cudaStreamDestroy(stream_));
src/cuda/cudaaligner.cpp:bool CUDABatchAligner::addOverlap(Overlap* overlap, std::vector<std::unique_ptr<Sequence>>& sequences)
src/cuda/cudaaligner.cpp:    // NOTE: The cudaaligner API for adding alignments is the opposite of edlib. Hence, what is
src/cuda/cudaaligner.cpp:    // treated as target in edlib is query in cudaaligner and vice versa.
src/cuda/cudaaligner.cpp:        fprintf(stderr, "Unknown error in cuda aligner!\n");
src/cuda/cudaaligner.cpp:void CUDABatchAligner::alignAll()
src/cuda/cudaaligner.cpp:void CUDABatchAligner::generate_cigar_strings()
src/cuda/cudaaligner.cpp:        throw std::runtime_error("Number of alignments doesn't match number of overlaps in cudaaligner.");
src/cuda/cudaaligner.cpp:void CUDABatchAligner::reset()

```
