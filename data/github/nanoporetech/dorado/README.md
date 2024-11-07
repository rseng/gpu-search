# https://github.com/nanoporetech/dorado

```console
DEV.md:Dorado requires CUDA 11.8 on Linux platforms. If the system you are running on does not have CUDA 11.8 installed, and you do not have sudo privileges, you can install locally from a run file as follows:
DEV.md:$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
DEV.md:$ sh cuda_11.8.0_520.61.05_linux.run --silent --toolkit --toolkitpath=${PWD}/cuda11.8
DEV.md:In this case, cmake should be invoked with `CUDAToolkit_ROOT` in order to tell the build process where to find CUDA:
DEV.md:$ cmake -DCUDAToolkit_ROOT=~/dorado_deps/cuda11.8 -S . -B cmake-build
DEV.md:Note that a [suitable NVIDIA driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id3) will be required in order to run Dorado.
documentation/SAM.md:@PG     ID:basecaller   PN:dorado       VN:0.2.4+3fc2b0f        CL:dorado basecaller dna_r10.4.1_e8.2_400bps_hac@v4.1.0 pod5/        DS:gpu:Quadro GV100
CHANGELOG.md:* 9a81cbc269b742cad64b514a50aeb30a7bf25811 - Fix errors when running `dorado basecaller` on multi-GPU systems
CHANGELOG.md:* 7f40154aed13324865123386b278f3782eafa107 - Fix bug when running on GPUs with less than the recommended VRAM
CHANGELOG.md:This release of Dorado includes fixes and improvements to the Dorado 0.8.0 release, including corrected configuration for DNA v5 SUP-compatible 5mC_5hmC and 5mCG_5hmCG models, improved cDNA poly(A) tail estimation for data from MinION flow cells, reduced basecaller startup time on supported GPUs, and more.
CHANGELOG.md:* 762e88689b31099081a7fcd39b959ddc4c7eb2e1 - Cache batch sizes to significantly reduce basecaller startup time on supported GPUs
CHANGELOG.md:This release of Dorado adds v5.1 RNA models with new `inosine_m6A` and `m5C` RNA modified base models, updates existing modified base models, improves the speed of v5 SUP basecalling models on A100/H100 GPUs, and enhances the flexibility and stability of `dorado correct`. It also introduces per-barcode configuration for poly(A) estimation with interrupted tails, adds new `--output-dir` and `--bed-file` arguments to Dorado basecalling commands, and includes a variety of other improvements for stability and usability.
CHANGELOG.md:* 8e3a8707be5248d7bcc47d3e89b80c0bdc9c2f36 - Improve speed of v5 SUP basecalling models on A100 and H100 GPUs
CHANGELOG.md:* 5ddfc2fa6d639fa52c735184a2c92e9ce4306a3c - Fix bug causing CUDA illegal memory access with v5 RNA SUP and mods
CHANGELOG.md:* 45b8acc730ddbe6b438cf17153a64c084d1af2fb - Package missing CUDA Toolkit dependencies with `dorado`
CHANGELOG.md:* 580ad61ccd0f193c88202c852cbea38790c50700 - Prevent creation of CUDA stream when device is CPU
CHANGELOG.md:* 298277150ad2522ca6c1928c4981782ce2893a5a - Fix issue with allocating memory on unused GPU during basecalling
CHANGELOG.md:This release of Dorado improves performance for short read basecalling and RBK barcode classification rates, introduces sorted and indexed BAM generation in Dorado aligner and demux, and updates the minimap2 version and default mapping preset. It also adds GPU information to the output BAM or FASTQ and includes several other improvements and bug fixes.
CHANGELOG.md:* 913f062d6efac4e661bea0142c644331a01a5c29 - Add DS:gpu information to output FASTQ and SAM/BAM files
CHANGELOG.md:* ec106d65eed2592683a8d30f936318721369b427 - Improve performance of calling modified bases on NVIDIA GPUs
CHANGELOG.md:* 6f283a5f6876df767b09ee4c4ec4d6e268731b9d - Prevent CUDA OOM due to small allocations
CHANGELOG.md:* 0fa2c2f7f964942287548ecc3afc439a9c530b5c - Fix Cuda OOM during batch size calculation
CHANGELOG.md:* 901f700e00c76158a11aa4fa34425c107fa15a18 - Improve error reporting when the device string is invalid for CUDA devices
CHANGELOG.md:* 6ed81c5edcf0d6e0d20da4d250891e0a7e77b92a - Run separate modbase models in different CUDA streams.
CHANGELOG.md:* d92547a49fb95210656fcef7e9486f228ab4cfeb - Make CUDA kernel profiling to stderr available via `--devopts`
CHANGELOG.md:* f14b418e8497fcb033d68f3e6561b5b9a91a07fb - Provide NVIDIA driver version in server API
CHANGELOG.md:* 207871ee7f98d1e0cbefeb790ffdd02c1f6c929c - Use `DORADO_GPU_BUILD` rather than `!defined(__x86_64__)`
CHANGELOG.md:* d16ccbea8d99c1d820d508ee3ce9175374293049 - Get tests that use CUDA working when ASAN is enabled
CHANGELOG.md:* 8ec58f0153b39f9e60d0606da0a4cc24c6d83e59 - Option for `--guard-gpus` no longer used in duplex
CHANGELOG.md:* 1f3cade20f6074880997833783ae1be645974557 - Add `CUDAGuard` before cache clear to reduce CUDA memory consumption
CHANGELOG.md:This release of Dorado contains a few bug fixes and a hotfix for CUDA out of memory issues encountered during duplex runs with v0.3.3.
CHANGELOG.md: * f6a0422b34cdd89637bddb939364a38ea507d198 - Fix CUDA OOM in duplex by removing tensor caching in decode and updating memory fraction for stereo model.
CHANGELOG.md: * 7307146b55096ee339b1da953b5acf05d112e124 - Major reduction to required GPU memory, especially for A100/H100. Allows greater batch size and consequently improved basecalling speed
CHANGELOG.md: * 09c5b28f51707c72df1436967d5da28f850beef1 - Speed up the fixed cost of auto batchsize detection on mGPU systems by running in parallel
CHANGELOG.md: * 24b6c4e2854c32aaca37b039c7bdf5a773db8995 - Retry basecalling on CUDA OOM after clearing allocator cache
CHANGELOG.md: * 7e70de7197ffc09586fe6f097f9b25c30dc17802 - Add support for compute 6.1 (GTX 1080 Ti) plus handling CUDA failures
CHANGELOG.md: * 790a002be6e69de390ca9d4259e126a02c0beafa - Mitigate simplex scaling performance regression on mGPU systems.
CHANGELOG.md:This is a major release of Dorado which introduces: Duplex pairing and splitting for directly going from POD5 to duplex reads, major performance improvements to simplex and duplex basecalling on A100 GPUs via int8 model quantization and the output of aligned BAM from Dorado and support for producing summary tsv files from BAM.
CHANGELOG.md: * 4d91533610ca908dd4daf61a81e3a3fe634ace89 - Improvements to reduce possibility of out of memory issues on CUDA devices via a GPU device mutex
CHANGELOG.md: * 98eb30d3a23c0e40121d030cb73be38a045dbfa2 - Add Cutlass LSTM kernels for significant performance improvement on A100 GPUs
CHANGELOG.md: * 1079b75303ddf642a6569141c7cdcbfd808e6826 - Upgrade to Torch 2.0 and Cuda 11.8
CHANGELOG.md: * 3e3b21a4c5e571ee420e8b37cbafd42aa6b7fd5a - Remove deprecated use of FindCUDA and show real location of found toolkit
CHANGELOG.md: * 824459e4f4b8a7fa4c160c1af76d2a5ef760c66f - Add `"cuda:auto"` as alternative to `"cuda:all"` when selecting a compute accelerator device on CLI
CHANGELOG.md: * 6b9249f4cc64ecb43134239fba2fe5682c5deb72 - Initial CUDA 12.0 support
CHANGELOG.md:Release of version 0.0.2 is a minor release which introduces several performance and usability improvements to Dorado. In particular, we are happy to announce the inclusion of multi-GPU base modification calling and 5mC models.
CHANGELOG.md: * 626137a2fae13b8ac641fc7c032bb0ae150cbfad - Improve the way we look up GPU core count on Apple
CHANGELOG.md: * e9c78665ed8cf4d76de5a0fb57cd19fdd03c5a44 - Reduced Startup cost on multi-GPU systems
CHANGELOG.md: * bd6014edc8de374645ade284dd103eccbfa481db - Support for basecalling on systems with multiple Nvidia GPUs.
CHANGELOG.md: * 6ec50dc5cecc65f0ff940420c0de152ba561f85c - Major rearchitecture of CUDA model runners for  higher basecalling speed and lower GPU memory utilisation.
tests/gpu_monitor_test.cpp:#include "torch_utils/gpu_monitor.h"
tests/gpu_monitor_test.cpp:#define CUT_TAG "[dorado::utils::gpu_monitor]"
tests/gpu_monitor_test.cpp:    TEST_CASE_METHOD(GpuMonitorTestFixture, CUT_TAG " " name, CUT_TAG)
tests/gpu_monitor_test.cpp:class GpuMonitorTestFixture {
tests/gpu_monitor_test.cpp:    GpuMonitorTestFixture() {
tests/gpu_monitor_test.cpp:        num_devices = dorado::utils::gpu_monitor::get_device_count();
tests/gpu_monitor_test.cpp:            if (dorado::utils::gpu_monitor::detail::is_accessible_device(index)) {
tests/gpu_monitor_test.cpp:namespace dorado::utils::gpu_monitor::test {
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_nvidia_driver_version has value if torch::hasCUDA") {
tests/gpu_monitor_test.cpp:    auto driver_version = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:    if (torch::hasCUDA()) {
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_nvidia_driver_version retruns valid version string") {
tests/gpu_monitor_test.cpp:    auto driver_version = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_nvidia_driver_version multiple calls return the same result") {
tests/gpu_monitor_test.cpp:    auto driver_version_0 = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:    auto driver_version_1 = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_nvidia_driver_version does not have value on Apple") {
tests/gpu_monitor_test.cpp:    auto driver_version = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_nvidia_driver_version always has a value on Jetson") {
tests/gpu_monitor_test.cpp:    auto driver_version = get_nvidia_driver_version();
tests/gpu_monitor_test.cpp:DEFINE_TEST("parse_nvidia_version_line parameterised test") {
tests/gpu_monitor_test.cpp:                    "NVRM version: NVIDIA UNIX x86_64 Kernel Module  520.61.05  Thu Sep 29 "
tests/gpu_monitor_test.cpp:                    "NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  378.13  Release "
tests/gpu_monitor_test.cpp:                    "NVRM version: NVIDIA UNIX Kernel Module for aarch64  34.1.1  Release Build  "
tests/gpu_monitor_test.cpp:        auto version = detail::parse_nvidia_version_line(test.line);
tests/gpu_monitor_test.cpp:DEFINE_TEST("parse_nvidia_tegra_line parameterised test") {
tests/gpu_monitor_test.cpp:        auto version = detail::parse_nvidia_tegra_line(test.line);
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    CAPTURE(info->gpu_shutdown_temperature_error);
tests/gpu_monitor_test.cpp:    REQUIRE(info->gpu_shutdown_temperature.has_value());
tests/gpu_monitor_test.cpp:    CHECK(*info->gpu_shutdown_temperature > 0);
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    CAPTURE(info->gpu_slowdown_temperature_error);
tests/gpu_monitor_test.cpp:    REQUIRE(info->gpu_slowdown_temperature.has_value());
tests/gpu_monitor_test.cpp:    CHECK(*info->gpu_slowdown_temperature > 0);
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    CAPTURE(info->gpu_max_operating_temperature_error);
tests/gpu_monitor_test.cpp:    REQUIRE(info->gpu_max_operating_temperature.has_value());
tests/gpu_monitor_test.cpp:    CHECK(*info->gpu_max_operating_temperature > 0);
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:        "get_device_status_info with valid device returns with percentage_utilization_gpu in "
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    REQUIRE(info->percentage_utilization_gpu.has_value());
tests/gpu_monitor_test.cpp:    CHECK(*info->percentage_utilization_gpu <= 100);
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:    // N.B. test may fail has_value() check if a CI runner GPU does not support an nvml query
tests/gpu_monitor_test.cpp:DEFINE_TEST("get_device_count returns a non zero value if torch getNumGPUs is non-zero") {
tests/gpu_monitor_test.cpp:    if (!torch::getNumGPUs()) {
tests/gpu_monitor_test.cpp:}  // namespace dorado::utils::gpu_monitor::test
tests/cuda_utils_test.cpp:#include "torch_utils/cuda_utils.h"
tests/cuda_utils_test.cpp:#define CUT_TAG "[cuda_utils]"
tests/cuda_utils_test.cpp:namespace dorado::utils::cuda_utils {
tests/cuda_utils_test.cpp:    if (!torch::hasCUDA()) {
tests/cuda_utils_test.cpp:        spdlog::warn("No Nvidia driver present - Test skipped");
tests/cuda_utils_test.cpp:    auto options = at::TensorOptions().dtype(torch::kFloat16).device(c10::kCUDA);
tests/cuda_utils_test.cpp:                    {"cuda:all", 1, true, {0}},
tests/cuda_utils_test.cpp:                    {"cuda:all", 0, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:all", 4, true, {0, 1, 2, 3}},
tests/cuda_utils_test.cpp:                    {"cuda:2", 2, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:-1", 1, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:2", 3, true, {2}},
tests/cuda_utils_test.cpp:                    {"cuda:2,0,3", 4, true, {0, 2, 3}},
tests/cuda_utils_test.cpp:                    {"cuda:0,0", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:0,1,2,1", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:a", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:a,0", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:0,a", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:1-3", 4, false, {}},
tests/cuda_utils_test.cpp:                    {"cuda:1.3", 4, false, {}},
tests/cuda_utils_test.cpp:}  // namespace dorado::utils::cuda_utils
tests/ScaledDotProductAttention.cpp:#if DORADO_CUDA_BUILD
tests/ScaledDotProductAttention.cpp:            c10::kCUDA,
tests/ScaledDotProductAttention.cpp:#endif  // DORADO_CUDA_BUILD
tests/ScaledDotProductAttention.cpp:    if ((device_type == c10::kCUDA && !torch::hasCUDA()) ||
tests/data/aligner_test/prealigned.sam:@PG	ID:basecaller	PN:dorado	VN:0.7.2+9ac85c6	CL:dorado basecaller /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 /mmfs1/groups/datasets/active/flowcells/dna_r10.4.1_e8.2.1_2d_prom_400_5kHz/hemimethyl_controls/20240711_SS_KL_2d-hemi-meth/2D_FC1/20240711_1617_6B_PAY97752_8e76b683/pod5 --emit-moves --no-trim --modified-bases-models /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v1 --max-reads 5000000	DS:gpu:NVIDIA A100-PCIE-40GB
tests/data/aligner_test/prealigned.sam:@PG	ID:basecaller-7F2AA5CF	PN:dorado	VN:0.7.2+9ac85c6	CL:dorado basecaller /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 /mmfs1/groups/datasets/active/flowcells/dna_r10.4.1_e8.2.1_2d_prom_400_5kHz/hemimethyl_controls/20240711_SS_KL_2d-hemi-meth/2D_FC2/20240711_1617_6C_PAY96174_4d5fcfbb/pod5 --emit-moves --no-trim --modified-bases-models /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v1 --max-reads 5000000	DS:gpu:NVIDIA A100-PCIE-40GB
tests/data/aligner_test/prealigned.sam:@PG	ID:basecaller-29F37DCA	PN:dorado	VN:0.7.2+9ac85c6	CL:dorado basecaller /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0 /mmfs1/groups/datasets/active/flowcells/dna_r10.4.1_e8.2.1_2d_prom_400_5kHz/hemimethyl_controls/20240711_SS_KL_2d-hemi-meth/2D_FC3/20240711_1617_6D_PAY96132_f689e234/pod5 --emit-moves --no-trim --modified-bases-models /mmfs1/groups/machine_learning/active/arand/projects/analysis-notebooks/2d-remora/remora_on_complement_eda/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.0.0_5mC_5hmC@v1 --max-reads 5000000	DS:gpu:NVIDIA A100 80GB PCIe
tests/MetalLinearTest.cpp:// 32 threads/SIMD group on Apple GPUs.  The kernels have this hardwired.
tests/MetalLinearTest.cpp:    // get_mtl_device sets up an allocator that provides GPU/CPU shared memory
tests/MetalLinearTest.cpp:    // This equates to the number of GPU cores.  16 is the figure for a complete M1 Pro.
tests/MetalLinearTest.cpp:                        auto out_gpu_f32 = torch::zeros({lstm_chunk_size, in_batch_size, out_size},
tests/MetalLinearTest.cpp:                            auto out_gpu_partial = torch::zeros(
tests/MetalLinearTest.cpp:                                     mtl_for_tensor(out_gpu_partial)},
tests/MetalLinearTest.cpp:                            out_gpu_f32.slice(1, in_batch_offset,
tests/MetalLinearTest.cpp:                                              in_batch_offset + out_batch_size) = out_gpu_partial;
tests/MetalLinearTest.cpp:                        // These tolerances are somewhat arbitary, but we must account for GPU calculations in float16
tests/MetalLinearTest.cpp:                        auto out_gpu_2d = out_gpu_f32.view({-1, out_size});
tests/MetalLinearTest.cpp:                        CHECK(torch::allclose(out_cpu, out_gpu_2d, kRelTolerance, kAbsTolerance));
tests/MetalLinearTest.cpp:                        CHECK(MeanAbsDiff(out_cpu, out_gpu_2d) < kMeanAbsDiffTolerance);
tests/CMakeLists.txt:    gpu_monitor_test.cpp
tests/CMakeLists.txt:if (DORADO_GPU_BUILD)
tests/CMakeLists.txt:        target_sources(dorado_tests PRIVATE cuda_utils_test.cpp)
tests/fastq_reader_test.cpp:             "basecall_gpu=NVIDIA RTX A5500 Laptop GPU",
tests/NodeSmokeTest.cpp:#include <torch/cuda.h>
tests/NodeSmokeTest.cpp:#if DORADO_CUDA_BUILD
tests/NodeSmokeTest.cpp:#include "torch_utils/cuda_utils.h"
tests/NodeSmokeTest.cpp:    auto gpu = GENERATE(true, false);
tests/NodeSmokeTest.cpp:    CAPTURE(gpu);
tests/NodeSmokeTest.cpp:    if (gpu) {
tests/NodeSmokeTest.cpp:#elif DORADO_CUDA_BUILD
tests/NodeSmokeTest.cpp:        device = "cuda:all";
tests/NodeSmokeTest.cpp:        if (!dorado::utils::try_parse_cuda_device_string(device, devices, error_message) ||
tests/NodeSmokeTest.cpp:            SKIP("No CUDA devices found: " << error_message);
tests/NodeSmokeTest.cpp:        SKIP("Can't test GPU without DORADO_GPU_BUILD");
tests/NodeSmokeTest.cpp:    auto gpu = GENERATE(true, false);
tests/NodeSmokeTest.cpp:    CAPTURE(gpu);
tests/NodeSmokeTest.cpp:    if (gpu) {
tests/NodeSmokeTest.cpp:#elif DORADO_CUDA_BUILD
tests/NodeSmokeTest.cpp:        device = "cuda:all";
tests/NodeSmokeTest.cpp:        if (!dorado::utils::try_parse_cuda_device_string(device, devices, error_message) ||
tests/NodeSmokeTest.cpp:            SKIP("No CUDA devices found: " << error_message);
tests/NodeSmokeTest.cpp:        SKIP("Can't test GPU without DORADO_GPU_BUILD");
tests/symbol_test.cpp:#include "torch_utils/gpu_monitor.h"
tests/symbol_test.cpp:#elif DORADO_CUDA_BUILD
tests/symbol_test.cpp:#include "torch_utils/cuda_utils.h"
tests/symbol_test.cpp:#if DORADO_CUDA_BUILD
tests/symbol_test.cpp:    // torch_utils/cuda_utils.h
tests/symbol_test.cpp:    force_reference(&dorado::utils::acquire_gpu_lock);
tests/symbol_test.cpp:    // torch_utils/gpu_monitor.h
tests/symbol_test.cpp:    force_reference(&dorado::utils::gpu_monitor::get_device_count);
README.md:* Runs on Apple silicon (M1/2 family) and Nvidia GPUs including multi-GPU with linear scaling (see [Platforms](#platforms)).
README.md:* Multiple custom optimisations in CUDA and Metal for maximising inference performance.
README.md:Dorado is heavily-optimised for Nvidia A100 and H100 GPUs and will deliver maximal performance on systems with these GPUs.
README.md:| Platform | GPU/CPU | Minimum Software Requirements |
README.md:| Linux x86_64  | (G)V100, A100 | CUDA Driver ≥450.80.02 |
README.md:| | H100 | CUDA Driver ≥520 |
README.md:| Windows x86_64 | (G)V100, A100 | CUDA Driver ≥452.39 |
README.md:| | H100 | CUDA Driver ≥520 |
README.md:Linux or Windows systems not listed above but which have Nvidia GPUs with ≥8 GB VRAM and architecture from Pascal onwards (except P100/GP100) have not been widely tested but are expected to work. When basecalling with Apple devices, we recommend systems with ≥16 GB of unified memory.
README.md:AWS Benchmarks on Nvidia GPUs for Dorado 0.3.0 are available [here](https://aws.amazon.com/blogs/hpc/benchmarking-the-oxford-nanopore-technologies-basecallers-on-aws/). Please note: Dorado's basecalling speed is continuously improving, so these benchmarks may not reflect performance with the latest release.
README.md:2. Dorado will automatically detect your GPU's free memory and select an appropriate batch size.
README.md:3. Dorado will automatically run in multi-GPU `cuda:all` mode. If you have a hetrogenous collection of GPUs, select the faster GPUs using the `--device` flag (e.g `--device cuda:0,2`). Not doing this will have a detrimental impact on performance.
README.md:The error correction tool is both compute and memory intensive. As a result, it is best run on a system with multiple high performance CPU cores ( >= 64 cores), large system memory ( >= 256GB) and a modern GPU with a large VRAM ( >= 32GB).
README.md:Dorado Correct now also provides a feature to run mapping (CPU-only stage) and inference (GPU-intensive stage) individually. This enables separation of the CPU and GPU heavy stages into individual steps which can even be run on different nodes with appropriate compute characteristics. Example:
README.md:2. The auto-computed inference batch size may still be too high for your system. If you are experiencing warnings/errors regarding available GPU memory, try reducing the batch size / selecting it manually. For example:
README.md:Dorado comes equipped with the necessary libraries (such as CUDA) for its execution. However, on some operating systems, the system libraries might be chosen over Dorado's. This discrepancy can result in various errors, for instance,  `CuBLAS error 8`.
README.md:### GPU Out of Memory Errors
README.md:Dorado operates on a broad range of GPUs but it is primarily developed for Nvidia A100/H100 and Apple Silicon. Dorado attempts to find the optimal batch size for basecalling. Nevertheless, on some low-RAM GPUs, users may face out of memory crashes.
README.md:### Low GPU Utilization
README.md:Low GPU utilization can lead to reduced basecalling speed. This problem can be identified using tools such as `nvidia-smi` and `nvtop`. Low GPU utilization often stems from I/O bottlenecks in basecalling. Here are a few steps you can take to improve the situation:
.gitmodules:	url = https://github.com/NVIDIA/NVTX.git
CMakeLists.txt:    add_compile_options("$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/MP${WIN_THREADS}>")
CMakeLists.txt:# We don't support GPU builds on macOS/x64
CMakeLists.txt:    set(DORADO_GPU_BUILD FALSE)
CMakeLists.txt:    set(DORADO_GPU_BUILD TRUE)
CMakeLists.txt:find_package(CUDAToolkit QUIET)
CMakeLists.txt:if(${CUDAToolkit_FOUND})
CMakeLists.txt:  file(REAL_PATH ${CUDAToolkit_TARGET_DIR} CUDAToolkit_REAL_DIR)
CMakeLists.txt:  message(STATUS "Found CUDA ${CUDAToolkit_VERSION} (${CUDAToolkit_TARGET_DIR} -> ${CUDAToolkit_REAL_DIR})")
CMakeLists.txt:if (DORADO_GPU_BUILD AND APPLE)
CMakeLists.txt:    if((CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64*|^arm*") AND (${CUDAToolkit_VERSION} VERSION_LESS 11.0))
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_fast_cuda0_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:0",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_hac_cuda0_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:0",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_sup_cuda0_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:0",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_hac_cuda0_5mCG-5hmCG_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:0",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_fast_cudaall_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:all",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_hac_cudaall_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:all",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_sup_cudaall_nomods_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:all",
benchmark/benchmark.py:    "dna_kit14_400bps_5khz_simplex_hac_cudaall_5mCG-5hmCG_noaln_nobarcode": DoradoConfig(
benchmark/benchmark.py:        devices="cuda:all",
benchmark/benchmark.py:    gpu_type: str,
benchmark/benchmark.py:            "type,duplex,model_variant,devices,max_reads,mods,reference,barcode_kit,platform,gpu_type,sequencer,speed\n"
benchmark/benchmark.py:            f"{config.data_type},{config.duplex},{config.model_variant},{config.devices},{config.max_reads},{config.mods},{config.reference},{config.barcode_kit},{platform},{gpu_type},{sequencer},{speed}\n"
benchmark/benchmark.py:        "--gpu-type", help="Type of GPU being used", type=str, required=True
benchmark/benchmark.py:            args.gpu_type,
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:# Extract the GPU name from the benchmark filenames
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:gpu_name="${benchmark_files[0]}"
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:gpu_name=${gpu_name#*__}
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:gpu_name=${gpu_name%%__*}
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:gpu_name="${gpu_name// /_}"
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:gpu_name_no_dashes="${gpu_name//-/_}"
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:echo gpu name is: $gpu_name
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:rm ${gpu_name}.cpp || true
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:rm ${gpu_name}.h || true
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:echo "#include \"${gpu_name}.h\"
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:void Add${gpu_name_no_dashes}Benchmarks(std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>& chunk_benchmarks) {" >> ${gpu_name}.cpp
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:cat chunk_benchmarks__*.txt >> ${gpu_name}.cpp
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:" >> ${gpu_name}.cpp
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:    void Add${gpu_name_no_dashes}Benchmarks(std::map<std::pair<std::string, std::string>, std::unordered_map<int, float>>& chunk_benchmarks);
benchmark/generate_chunk_auto_batchsize_benchmarks.sh:" >> ${gpu_name}.h
cmake/InstallRedistLibs.cmake:    # CUDA toolkit DLLs we depend on:
cmake/InstallRedistLibs.cmake:    set(VERSIONED_CUDA_LIBS
cmake/InstallRedistLibs.cmake:        libcudart*.so*
cmake/InstallRedistLibs.cmake:        list(APPEND VERSIONED_CUDA_LIBS
cmake/InstallRedistLibs.cmake:    foreach(LIB IN LISTS VERSIONED_CUDA_LIBS)
cmake/InstallRedistLibs.cmake:        # torch may bundle it's own specific copy of the cuda libs. if it does, we want everything to point at them
cmake/InstallRedistLibs.cmake:        file(GLOB TORCH_CUDA_LIBS "${TORCH_LIB}/lib/${LIB}")
cmake/InstallRedistLibs.cmake:        if(TORCH_CUDA_LIBS)
cmake/InstallRedistLibs.cmake:            list(SORT TORCH_CUDA_LIBS)
cmake/InstallRedistLibs.cmake:            foreach(TORCH_CUDA_LIB IN LISTS TORCH_CUDA_LIBS)
cmake/InstallRedistLibs.cmake:                set(target ${TORCH_CUDA_LIB})
cmake/InstallRedistLibs.cmake:            # bundle the libraries from the cuda toolkit
cmake/InstallRedistLibs.cmake:            file(GLOB NATIVE_CUDA_LIBS "${CUDAToolkit_TARGET_DIR}/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/lib/${LIB}")
cmake/InstallRedistLibs.cmake:            install(FILES ${NATIVE_CUDA_LIBS} DESTINATION lib COMPONENT redist_libs)
cmake/Koi.cmake:function(get_best_compatible_koi_version KOI_CUDA)
cmake/Koi.cmake:        if (${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL ${SUPPORTED_VERSION})
cmake/Koi.cmake:            set(${KOI_CUDA} ${SUPPORTED_VERSION} PARENT_SCOPE)
cmake/Koi.cmake:    message(FATAL_ERROR "Unsupported CUDA toolkit version: ${CUDAToolkit_VERSION}")
cmake/Koi.cmake:        find_package(CUDAToolkit REQUIRED)
cmake/Koi.cmake:        get_best_compatible_koi_version(KOI_CUDA)
cmake/Koi.cmake:        set(KOI_DIR libkoi-${KOI_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}-cuda-${KOI_CUDA})
cmake/OpenSSL.cmake:            if(${CUDAToolkit_VERSION} VERSION_LESS 11.0)
cmake/Torch.cmake:    find_package(CUDAToolkit REQUIRED)
cmake/Torch.cmake:    # the torch cuda.cmake will set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}") [2]
cmake/Torch.cmake:    # so we need to make CUDA_TOOLKIT_ROOT_DIR is set correctly as per [1]
cmake/Torch.cmake:    # 1. https://cmake.org/cmake/help/latest/module/FindCUDAToolkit.html
cmake/Torch.cmake:    # 2. https://github.com/pytorch/pytorch/blob/5fa71207222620b4efb78989849525d4ee6032e8/cmake/public/cuda.cmake#L40
cmake/Torch.cmake:    if(DEFINED CUDAToolkit_ROOT)
cmake/Torch.cmake:      set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT})
cmake/Torch.cmake:        # Bodge for Torch, since static linking assumes find_package(CUDA) has already been called
cmake/Torch.cmake:        find_package(CUDA REQUIRED)
cmake/Torch.cmake:    if(NOT DEFINED CMAKE_CUDA_COMPILER)
cmake/Torch.cmake:      if(DEFINED CUDAToolkit_ROOT)
cmake/Torch.cmake:        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
cmake/Torch.cmake:        set(CMAKE_CUDA_COMPILER ${CUDAToolkit_NVCC_EXECUTABLE})
cmake/Torch.cmake:    # https://github.com/pytorch/pytorch/blob/7289d22d6749465d3bae2cb5a6ce04729318f55b/cmake/public/cuda.cmake#L173
cmake/Torch.cmake:    set(CMAKE_CUDA_ARCHITECTURES 62 70 72 75)
cmake/Torch.cmake:    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.3)
cmake/Torch.cmake:        list(APPEND CMAKE_CUDA_ARCHITECTURES 80 86)
cmake/Torch.cmake:    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.4)
cmake/Torch.cmake:      list(APPEND CMAKE_CUDA_ARCHITECTURES 87)
cmake/Torch.cmake:    if(${CUDAToolkit_VERSION} VERSION_GREATER_EQUAL 11.8)
cmake/Torch.cmake:      list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
cmake/Torch.cmake:            if(${CUDAToolkit_VERSION} VERSION_LESS 11.0)
cmake/Torch.cmake:                    # Grab from NVidia rather than pytorch so that it has the magic NVidia sauce
cmake/Torch.cmake:                    set(TORCH_URL https://developer.download.nvidia.com/compute/redist/jp/v502/pytorch/torch-1.13.0a0+d0d6b1f2.nv22.09-cp38-cp38-linux_aarch64.whl)
cmake/Torch.cmake:        # Note we need to use the generator expression to avoid setting this for CUDA.
cmake/Torch.cmake:        set(PYTORCH_BUILD_VERSION "import torch; print('%s+cu%s' % (torch.__version__, torch.version.cuda.replace('.', '')), end='')")
cmake/Torch.cmake:            CUDA::cudart_static
cmake/Torch.cmake:            CUDA::cublas
cmake/Torch.cmake:            CUDA::cufft
cmake/Torch.cmake:            CUDA::cusolver
cmake/Torch.cmake:            CUDA::cusparse
cmake/Torch.cmake:    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND ${CUDAToolkit_VERSION} VERSION_LESS 11.0)
cmake/Torch.cmake:            # Some of the CUDA libs have inter-dependencies, so group them together
cmake/Torch.cmake:                CUDA::cudart_static,
cmake/Torch.cmake:                CUDA::cublas_static,
cmake/Torch.cmake:                CUDA::cublasLt_static,
cmake/Torch.cmake:                CUDA::cufft_static_nocallback,
cmake/Torch.cmake:                CUDA::cusolver_static,
cmake/Torch.cmake:                ${CUDAToolkit_TARGET_DIR}/lib64/liblapack_static.a,
cmake/Torch.cmake:                CUDA::cusparse_static,
cmake/Torch.cmake:                CUDA::cupti,
cmake/Torch.cmake:                CUDA::curand_static,
cmake/Torch.cmake:                CUDA::nvrtc,
cmake/Torch.cmake:                CUDA::culibos
cmake/Torch.cmake:        # Some CUDA lib symbols have internal linkage, so they must be part of the helper lib too
cmake/Torch.cmake:        set(ont_cuda_internal_linkage_libs CUDA::culibos CUDA::cudart_static)
cmake/Torch.cmake:        if (TARGET CUDA::cupti_static)
cmake/Torch.cmake:            list(APPEND ont_cuda_internal_linkage_libs CUDA::cupti_static)
cmake/Torch.cmake:        elseif(TARGET CUDA::cupti)
cmake/Torch.cmake:            # CUDA::cupti appears to be static if CUDA::cupti_static doesn't exist
cmake/Torch.cmake:            list(APPEND ont_cuda_internal_linkage_libs CUDA::cupti)
cmake/Torch.cmake:        elseif(EXISTS ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64/libcupti_static.a)
cmake/Torch.cmake:            list(APPEND ont_cuda_internal_linkage_libs ${CUDAToolkit_TARGET_DIR}/extras/CUPTI/lib64/libcupti_static.a)
cmake/Torch.cmake:            set(ont_torch_extra_cuda_libs
cmake/Torch.cmake:                # I don't know why the MKL libs need to be part of the CUDA group, but having them in a
cmake/Torch.cmake:                ${TORCH_LIB}/lib/libnccl_static.a
cmake/Torch.cmake:            set(ont_torch_extra_cuda_libs
cmake/Torch.cmake:                ${CUDAToolkit_TARGET_DIR}/lib64/liblapack_static.a
cmake/Torch.cmake:                CUDA::curand_static
cmake/Torch.cmake:                CUDA::nvrtc
cmake/Torch.cmake:                # Some of the CUDA libs have inter-dependencies, so group them together
cmake/Torch.cmake:                    ${ont_cuda_internal_linkage_libs}
cmake/Torch.cmake:            # Some of the CUDA libs have inter-dependencies, so group them together
cmake/Torch.cmake:                CUDA::cudart_static,
cmake/Torch.cmake:                CUDA::cublas_static,
cmake/Torch.cmake:                CUDA::cublasLt_static,
cmake/Torch.cmake:                CUDA::cufft_static_nocallback,
cmake/Torch.cmake:                CUDA::cusolver_static,
cmake/Torch.cmake:                CUDA::cusparse_static,
cmake/Torch.cmake:                ${ont_cuda_internal_linkage_libs},
cmake/Torch.cmake:                ${ont_torch_extra_cuda_libs}
cmake/Torch.cmake:        if (${CMAKE_VERSION} VERSION_LESS 3.23.4 AND EXISTS ${CUDAToolkit_TARGET_DIR}/lib64/libcusolver_lapack_static.a)
cmake/Torch.cmake:            # CUDA::cusolver_static is missing the cusolver_lapack_static target+dependency in older versions of cmake
cmake/Torch.cmake:                ${CUDAToolkit_TARGET_DIR}/lib64/libcusolver_lapack_static.a
cmake/Warnings.cmake:          $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W4 /WX /external:anglebrackets /external:W0>
cmake/Warnings.cmake:            $<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/W0>
dorado/cli/basecaller.cpp:#if DORADO_CUDA_BUILD
dorado/cli/basecaller.cpp:#include "torch_utils/cuda_utils.h"
dorado/cli/basecaller.cpp:#if DORADO_CUDA_BUILD
dorado/cli/basecaller.cpp:    auto initial_device_info = utils::get_cuda_device_info(device, false);
dorado/cli/basecaller.cpp:#if DORADO_CUDA_BUILD
dorado/cli/basecaller.cpp:        // We may have multiple GPUs with different amounts of free memory left after the modbase runners were created.
dorado/cli/basecaller.cpp:        // This allows us to set a different memory_limit_fraction in case we have a heterogeneous GPU setup
dorado/cli/basecaller.cpp:        auto updated_device_info = utils::get_cuda_device_info(device, false);
dorado/cli/basecaller.cpp:        std::vector<std::pair<std::string, float>> gpu_fractions;
dorado/cli/basecaller.cpp:            auto device_id = "cuda:" + std::to_string(updated_device_info[i].device_id);
dorado/cli/basecaller.cpp:            gpu_fractions.push_back(std::make_pair(device_id, fraction));
dorado/cli/basecaller.cpp:        cxxpool::thread_pool pool{gpu_fractions.size()};
dorado/cli/basecaller.cpp:        futures.reserve(gpu_fractions.size());
dorado/cli/basecaller.cpp:        for (const auto& [device_id, fraction] : gpu_fractions) {
dorado/cli/basecaller.cpp:            throw std::runtime_error("CUDA device requested but no devices found.");
dorado/cli/basecaller.cpp:    std::string gpu_names{};
dorado/cli/basecaller.cpp:#if DORADO_CUDA_BUILD
dorado/cli/basecaller.cpp:    gpu_names = utils::get_cuda_gpu_names(device);
dorado/cli/basecaller.cpp:    auto hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
dorado/cli/cli_utils.h:#if DORADO_CUDA_BUILD
dorado/cli/cli_utils.h:#include "torch_utils/cuda_utils.h"
dorado/cli/cli_utils.h:#if DORADO_CUDA_BUILD
dorado/cli/cli_utils.h:    auto gpu_string = utils::get_cuda_gpu_names(device);
dorado/cli/cli_utils.h:    if (!gpu_string.empty()) {
dorado/cli/cli_utils.h:        pg << "\tDS:gpu:" << gpu_string;
dorado/cli/cli_utils.h:        "Specify CPU or GPU device: 'auto', 'cpu', 'cuda:all' or "
dorado/cli/cli_utils.h:        "'cuda:<device_id>[,<device_id>...]'. Specifying 'auto' will choose either 'cpu', 'metal' "
dorado/cli/cli_utils.h:        "or 'cuda:all' depending on the presence of a GPU device."};
dorado/cli/cli_utils.h:#elif DORADO_CUDA_BUILD
dorado/cli/cli_utils.h:    if (!device.empty() && device.substr(0, 5) == "cuda:") {
dorado/cli/cli_utils.h:        if (utils::try_parse_cuda_device_string(device, devices, error_message)) {
dorado/cli/correct.cpp:#if DORADO_CUDA_BUILD
dorado/cli/duplex.cpp:#if DORADO_CUDA_BUILD
dorado/cli/duplex.cpp:#include "torch_utils/cuda_utils.h"
dorado/cli/duplex.cpp:    // on smaller VRAM GPUs is fixed.
dorado/cli/duplex.cpp:    // the CUDACaching allocator since the workspace memory is always cached
dorado/cli/duplex.cpp:    // simplex call is run on the same GPU, the allocator can't find enough
dorado/cli/duplex.cpp:        std::string gpu_names{};
dorado/cli/duplex.cpp:#if DORADO_CUDA_BUILD
dorado/cli/duplex.cpp:        gpu_names = utils::get_cuda_gpu_names(device);
dorado/cli/duplex.cpp:            hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
dorado/cli/duplex.cpp:            hts_writer = pipeline_desc.add_node<HtsWriter>({}, *hts_file, gpu_names);
dorado/cli/duplex.cpp:#if DORADO_CUDA_BUILD
dorado/cli/duplex.cpp:            auto initial_device_info = utils::get_cuda_device_info(device, false);
dorado/cli/duplex.cpp:#if DORADO_CUDA_BUILD
dorado/cli/duplex.cpp:                // We may have multiple GPUs with different amounts of free memory left after the modbase runners were created.
dorado/cli/duplex.cpp:                // This allows us to set a different memory_limit_fraction in case we have a heterogeneous GPU setup
dorado/cli/duplex.cpp:                auto updated_device_info = utils::get_cuda_device_info(device, false);
dorado/cli/duplex.cpp:                std::vector<std::pair<std::string, float>> gpu_fractions;
dorado/cli/duplex.cpp:                    auto device_id = "cuda:" + std::to_string(updated_device_info[i].device_id);
dorado/cli/duplex.cpp:                    gpu_fractions.push_back(std::make_pair(device_id, fraction));
dorado/cli/duplex.cpp:                cxxpool::thread_pool pool{gpu_fractions.size()};
dorado/cli/duplex.cpp:                    // The fraction argument for GPU memory allocates the fraction of the
dorado/cli/duplex.cpp:                    // WORKAROUND: As a workaround to CUDA OOM, force stereo to have a smaller
dorado/cli/duplex.cpp:                futures.reserve(gpu_fractions.size());
dorado/cli/duplex.cpp:                for (const auto& [device_id, fraction] : gpu_fractions) {
dorado/cli/duplex.cpp:                    throw std::runtime_error("CUDA device requested but no devices found.");
dorado/api/runner_creation.cpp:#elif DORADO_CUDA_BUILD
dorado/api/runner_creation.cpp:#include "basecall/CudaCaller.h"
dorado/api/runner_creation.cpp:#include "basecall/CudaModelRunner.h"
dorado/api/runner_creation.cpp:#include "torch_utils/cuda_utils.h"
dorado/api/runner_creation.cpp:        size_t num_gpu_runners,
dorado/api/runner_creation.cpp:    // Default is 1 device.  CUDA path may alter this.
dorado/api/runner_creation.cpp:        for (size_t i = 0; i < num_gpu_runners; i++) {
dorado/api/runner_creation.cpp:#if DORADO_CUDA_BUILD
dorado/api/runner_creation.cpp:        auto devices = dorado::utils::parse_cuda_device_string(params.device);
dorado/api/runner_creation.cpp:            throw std::runtime_error("CUDA device requested but no devices found.");
dorado/api/runner_creation.cpp:        std::vector<std::shared_ptr<basecall::CudaCaller>> callers;
dorado/api/runner_creation.cpp:        std::vector<std::future<std::shared_ptr<basecall::CudaCaller>>> futures;
dorado/api/runner_creation.cpp:            futures.push_back(pool.push(create_cuda_caller, per_device_params));
dorado/api/runner_creation.cpp:            for (size_t i = 0; i < num_gpu_runners; i++) {
dorado/api/runner_creation.cpp:                    runners.push_back(std::make_unique<basecall::CudaModelRunner>(callers[j],
dorado/api/runner_creation.cpp:    (void)num_gpu_runners;
dorado/api/runner_creation.cpp:#elif DORADO_CUDA_BUILD
dorado/api/runner_creation.cpp:        modbase_devices = dorado::utils::parse_cuda_device_string(device);
dorado/api/runner_creation.cpp:#if DORADO_CUDA_BUILD
dorado/api/runner_creation.cpp:size_t get_num_batch_dims(const std::shared_ptr<basecall::CudaCaller>& caller) {
dorado/api/runner_creation.cpp:basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller,
dorado/api/runner_creation.cpp:    return std::make_unique<basecall::CudaModelRunner>(std::move(caller), batch_dims_idx);
dorado/api/runner_creation.cpp:    return 1;  // Always 1 for Metal. Just needed for a unified interface for GPU builds.
dorado/api/caller_creation.h:#if DORADO_CUDA_BUILD
dorado/api/caller_creation.h:class CudaCaller;
dorado/api/caller_creation.h:#if DORADO_CUDA_BUILD
dorado/api/caller_creation.h:std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
dorado/api/runner_creation.h:        size_t num_gpu_runners,
dorado/api/runner_creation.h:#if DORADO_CUDA_BUILD
dorado/api/runner_creation.h:size_t get_num_batch_dims(const std::shared_ptr<basecall::CudaCaller>& caller);
dorado/api/runner_creation.h:basecall::RunnerPtr create_basecall_runner(std::shared_ptr<basecall::CudaCaller> caller,
dorado/api/caller_creation.cpp:#if DORADO_CUDA_BUILD
dorado/api/caller_creation.cpp:#include "basecall/CudaCaller.h"
dorado/api/caller_creation.cpp:#if DORADO_CUDA_BUILD
dorado/api/caller_creation.cpp:std::shared_ptr<basecall::CudaCaller> create_cuda_caller(
dorado/api/caller_creation.cpp:    return std::make_shared<basecall::CudaCaller>(params);
dorado/main.cpp:#include <cuda.h>
dorado/main.cpp:        std::cerr << "dorado:   " << DORADO_VERSION << "+cu" << CUDA_VERSION << '\n';
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/modbase/ModBaseCaller.cpp:#include <c10/cuda/CUDAStream.h>
dorado/modbase/ModBaseCaller.cpp:#include <torch/cuda.h>
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.cpp:    if (opts.device().is_cuda()) {
dorado/modbase/ModBaseCaller.cpp:        stream = c10::cuda::getStreamFromPool(false, opts.device().index());
dorado/modbase/ModBaseCaller.cpp:        c10::cuda::OptionalCUDAStreamGuard guard(stream);
dorado/modbase/ModBaseCaller.cpp:                        .pinned_memory(m_options.device().is_cuda())
dorado/modbase/ModBaseCaller.cpp:                        .pinned_memory(m_options.device().is_cuda())
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.cpp:    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());
dorado/modbase/ModBaseCaller.cpp:    c10::cuda::OptionalCUDAStreamGuard stream_guard(caller_data->stream);
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.cpp:        auto gpu_lock = [&] {
dorado/modbase/ModBaseCaller.cpp:            if (m_options.device().is_cuda()) {
dorado/modbase/ModBaseCaller.cpp:                return std::unique_lock(gpu_mutexes[m_options.device().index()]);
dorado/modbase/ModBaseCaller.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.h:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseCaller.h:#include <c10/cuda/CUDAStream.h>
dorado/modbase/ModBaseCaller.h:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.h:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.h:#include <c10/cuda/CUDAStream.h>
dorado/modbase/ModBaseRunner.h:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/modbase/ModBaseRunner.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.cpp:        if (caller->device().is_cuda()) {
dorado/modbase/ModBaseRunner.cpp:            streams.push_back(c10::cuda::getStreamFromPool(false, caller->device().index()));
dorado/modbase/ModBaseRunner.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.cpp:    // GPU base calling uses float16 signals and input tensors.
dorado/modbase/ModBaseRunner.cpp:#if DORADO_CUDA_BUILD
dorado/modbase/ModBaseRunner.cpp:    c10::cuda::OptionalCUDAStreamGuard guard(m_streams[model_id]);
dorado/basecall/CudaModelRunner.cpp:#include "CudaModelRunner.h"
dorado/basecall/CudaModelRunner.cpp:#include "CudaCaller.h"
dorado/basecall/CudaModelRunner.cpp:#include "torch_utils/cuda_utils.h"
dorado/basecall/CudaModelRunner.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/basecall/CudaModelRunner.cpp:CudaModelRunner::CudaModelRunner(std::shared_ptr<CudaCaller> caller, size_t batch_dims_idx)
dorado/basecall/CudaModelRunner.cpp:          m_stream(c10::cuda::getStreamFromPool(false, m_caller->device().index())) {
dorado/basecall/CudaModelRunner.cpp:void CudaModelRunner::accept_chunk(int chunk_idx, const at::Tensor &chunk) {
dorado/basecall/CudaModelRunner.cpp:std::vector<decode::DecodedChunk> CudaModelRunner::call_chunks(int num_chunks) {
dorado/basecall/CudaModelRunner.cpp:    c10::cuda::CUDAStreamGuard guard(m_stream);
dorado/basecall/CudaModelRunner.cpp:const CRFModelConfig &CudaModelRunner::config() const { return m_caller->config(); }
dorado/basecall/CudaModelRunner.cpp:size_t CudaModelRunner::chunk_size() const { return m_input.size(2); }
dorado/basecall/CudaModelRunner.cpp:size_t CudaModelRunner::batch_size() const { return m_input.size(0); }
dorado/basecall/CudaModelRunner.cpp:int CudaModelRunner::batch_timeout_ms() const { return m_caller->batch_timeout_ms(); }
dorado/basecall/CudaModelRunner.cpp:void CudaModelRunner::terminate() { m_caller->terminate(); }
dorado/basecall/CudaModelRunner.cpp:void CudaModelRunner::restart() { m_caller->restart(); }
dorado/basecall/CudaModelRunner.cpp:std::string CudaModelRunner::get_name() const {
dorado/basecall/CudaModelRunner.cpp:    name_stream << "CudaModelRunner_" << this;
dorado/basecall/CudaModelRunner.cpp:stats::NamedStats CudaModelRunner::sample_stats() const {
dorado/basecall/ModelRunner.cpp:          // TODO: m_options.dtype() depends on the device as TxModel uses kHalf in cuda which is not supported on CPU
dorado/basecall/CudaChunkBenchmarks.h:class CudaChunkBenchmarks final {
dorado/basecall/CudaChunkBenchmarks.h:    CudaChunkBenchmarks();
dorado/basecall/CudaChunkBenchmarks.h:    using GPUName = std::string;
dorado/basecall/CudaChunkBenchmarks.h:    std::map<std::pair<GPUName, ModelName>, ChunkTimings> m_chunk_benchmarks;
dorado/basecall/CudaChunkBenchmarks.h:            const GPUName& gpu_name,
dorado/basecall/CudaChunkBenchmarks.h:    static CudaChunkBenchmarks& instance() {
dorado/basecall/CudaChunkBenchmarks.h:        static CudaChunkBenchmarks chunk_benchmarks;
dorado/basecall/CudaChunkBenchmarks.h:    std::optional<const ChunkTimings> get_chunk_timings(const GPUName& gpu_name,
dorado/basecall/CudaChunkBenchmarks.h:    bool add_chunk_timings(const GPUName& gpu_name,
dorado/basecall/CudaCaller.h:#include <c10/cuda/CUDAStream.h>
dorado/basecall/CudaCaller.h:class CudaCaller {
dorado/basecall/CudaCaller.h:    CudaCaller(const BasecallerCreationParams &params);
dorado/basecall/CudaCaller.h:    ~CudaCaller();
dorado/basecall/CudaCaller.h:    std::string get_name() const { return std::string("CudaCaller_") + m_device; }
dorado/basecall/CudaCaller.h:        // TODO: we may want to use different numbers based on model type and GPU arch
dorado/basecall/CudaCaller.h:    void cuda_thread_fn();
dorado/basecall/CudaCaller.h:    std::thread m_cuda_thread;
dorado/basecall/CudaCaller.h:    c10::cuda::CUDAStream m_stream;
dorado/basecall/CudaCaller.h:    // A CudaCaller may accept chunks of multiple different sizes. Smaller sizes will be used to
dorado/basecall/decode/Decoder.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/decode/Decoder.cpp:#include "CUDADecoder.h"
dorado/basecall/decode/Decoder.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/decode/Decoder.cpp:    if (device.is_cuda()) {
dorado/basecall/decode/Decoder.cpp:        return std::make_unique<decode::CUDADecoder>(config.clamp ? 5.f : 0.f);
dorado/basecall/decode/CUDADecoder.h:class CUDADecoder final : public Decoder {
dorado/basecall/decode/CUDADecoder.h:    explicit CUDADecoder(float score_clamp_val) : m_score_clamp_val(score_clamp_val) {}
dorado/basecall/decode/CUDADecoder.h:    // We split beam_search into two parts, the first one running on the GPU and the second
dorado/basecall/decode/CUDADecoder.h:    // one on the CPU. While the second part is running we can submit more commands to the GPU
dorado/basecall/decode/CUDADecoder.cpp:#include "CUDADecoder.h"
dorado/basecall/decode/CUDADecoder.cpp:#include "torch_utils/cuda_utils.h"
dorado/basecall/decode/CUDADecoder.cpp:#include "torch_utils/gpu_profiling.h"
dorado/basecall/decode/CUDADecoder.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/basecall/decode/CUDADecoder.cpp:DecodeData CUDADecoder::beam_search_part_1(DecodeData data) const {
dorado/basecall/decode/CUDADecoder.cpp:    c10::cuda::CUDAGuard device_guard(scores.device());
dorado/basecall/decode/CUDADecoder.cpp:    utils::ScopedProfileRange loop{"gpu_decode", 1};
dorado/basecall/decode/CUDADecoder.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/decode/CUDADecoder.cpp:        dorado::utils::handle_cuda_result(host_back_guide_step(
dorado/basecall/decode/CUDADecoder.cpp:        dorado::utils::handle_cuda_result(host_beam_search_step(
dorado/basecall/decode/CUDADecoder.cpp:        dorado::utils::handle_cuda_result(host_compute_posts_step(
dorado/basecall/decode/CUDADecoder.cpp:        dorado::utils::handle_cuda_result(host_run_decode(
dorado/basecall/decode/CUDADecoder.cpp:std::vector<DecodedChunk> CUDADecoder::beam_search_part_2(DecodeData data) const {
dorado/basecall/CudaChunkBenchmarks.cpp:#include "CudaChunkBenchmarks.h"
dorado/basecall/CudaChunkBenchmarks.cpp:#include "benchmarks/NVIDIA_A100_80GB_PCIe.h"
dorado/basecall/CudaChunkBenchmarks.cpp:#include "benchmarks/NVIDIA_H100_PCIe.h"
dorado/basecall/CudaChunkBenchmarks.cpp:#include "benchmarks/NVIDIA_RTX_A6000.h"
dorado/basecall/CudaChunkBenchmarks.cpp:CudaChunkBenchmarks::CudaChunkBenchmarks() {
dorado/basecall/CudaChunkBenchmarks.cpp:    AddNVIDIA_A100_80GB_PCIeBenchmarks(m_chunk_benchmarks);
dorado/basecall/CudaChunkBenchmarks.cpp:    AddNVIDIA_H100_PCIeBenchmarks(m_chunk_benchmarks);
dorado/basecall/CudaChunkBenchmarks.cpp:    AddNVIDIA_RTX_A6000Benchmarks(m_chunk_benchmarks);
dorado/basecall/CudaChunkBenchmarks.cpp:std::optional<const CudaChunkBenchmarks::ChunkTimings>
dorado/basecall/CudaChunkBenchmarks.cpp:CudaChunkBenchmarks::get_chunk_timings_internal(const GPUName& gpu_name,
dorado/basecall/CudaChunkBenchmarks.cpp:    // Try looking up the specified gpu name directly
dorado/basecall/CudaChunkBenchmarks.cpp:    auto iter = m_chunk_benchmarks.find({gpu_name, model_name});
dorado/basecall/CudaChunkBenchmarks.cpp:    std::map<GPUName, GPUName> gpu_name_alias = {
dorado/basecall/CudaChunkBenchmarks.cpp:            {"NVIDIA A100-PCIE-40GB", "NVIDIA A100 80GB PCIe"},
dorado/basecall/CudaChunkBenchmarks.cpp:            {"NVIDIA A800 80GB PCIe", "NVIDIA A100 80GB PCIe"},
dorado/basecall/CudaChunkBenchmarks.cpp:    auto alias_name = gpu_name_alias.find(gpu_name);
dorado/basecall/CudaChunkBenchmarks.cpp:    if (alias_name != gpu_name_alias.cend()) {
dorado/basecall/CudaChunkBenchmarks.cpp:std::optional<const CudaChunkBenchmarks::ChunkTimings> CudaChunkBenchmarks::get_chunk_timings(
dorado/basecall/CudaChunkBenchmarks.cpp:        const GPUName& gpu_name,
dorado/basecall/CudaChunkBenchmarks.cpp:    return get_chunk_timings_internal(gpu_name, model_path);
dorado/basecall/CudaChunkBenchmarks.cpp:bool CudaChunkBenchmarks::add_chunk_timings(const GPUName& gpu_name,
dorado/basecall/CudaChunkBenchmarks.cpp:    if (get_chunk_timings_internal(gpu_name, model_name)) {
dorado/basecall/CudaChunkBenchmarks.cpp:    auto& new_benchmarks = m_chunk_benchmarks[{gpu_name, model_name}];
dorado/basecall/CudaModelRunner.h:#include <c10/cuda/CUDAStream.h>
dorado/basecall/CudaModelRunner.h:class CudaCaller;
dorado/basecall/CudaModelRunner.h:class CudaModelRunner final : public ModelRunnerBase {
dorado/basecall/CudaModelRunner.h:    explicit CudaModelRunner(std::shared_ptr<CudaCaller> caller, size_t batch_dims_idx);
dorado/basecall/CudaModelRunner.h:    std::shared_ptr<CudaCaller> m_caller;
dorado/basecall/CudaModelRunner.h:    c10::cuda::CUDAStream m_stream;
dorado/basecall/CudaCaller.cpp:#include "CudaCaller.h"
dorado/basecall/CudaCaller.cpp:#include "CudaChunkBenchmarks.h"
dorado/basecall/CudaCaller.cpp:#include "torch_utils/cuda_utils.h"
dorado/basecall/CudaCaller.cpp:#include <ATen/cuda/CUDAContext.h>
dorado/basecall/CudaCaller.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/basecall/CudaCaller.cpp:#include <torch/cuda.h>
dorado/basecall/CudaCaller.cpp:struct CudaCaller::NNTask {
dorado/basecall/CudaCaller.cpp:CudaCaller::CudaCaller(const BasecallerCreationParams &params)
dorado/basecall/CudaCaller.cpp:          m_stream(c10::cuda::getStreamFromPool(false, m_options.device().index())) {
dorado/basecall/CudaCaller.cpp:    assert(m_options.device().is_cuda());
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDAGuard device_guard(m_options.device());
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDACachingAllocator::emptyCache();
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
dorado/basecall/CudaCaller.cpp:CudaCaller::~CudaCaller() { terminate(); }
dorado/basecall/CudaCaller.cpp:std::vector<decode::DecodedChunk> CudaCaller::call_chunks(at::Tensor &input,
dorado/basecall/CudaCaller.cpp:void CudaCaller::terminate() {
dorado/basecall/CudaCaller.cpp:    if (m_cuda_thread.joinable()) {
dorado/basecall/CudaCaller.cpp:        m_cuda_thread.join();
dorado/basecall/CudaCaller.cpp:void CudaCaller::restart() {
dorado/basecall/CudaCaller.cpp:std::pair<at::Tensor, at::Tensor> CudaCaller::create_input_output_tensor(
dorado/basecall/CudaCaller.cpp:stats::NamedStats CudaCaller::sample_stats() const {
dorado/basecall/CudaCaller.cpp:std::pair<int64_t, int64_t> CudaCaller::calculate_memory_requirements() const {
dorado/basecall/CudaCaller.cpp:                    "Failed to set GPU memory requirements. Unexpected model out_features {}.",
dorado/basecall/CudaCaller.cpp:            spdlog::warn("Unexpected model out_features {}. Estimating GPU memory requirements.");
dorado/basecall/CudaCaller.cpp:            spdlog::error("Failed to set GPU memory requirements. Unexpected model insize {}.",
dorado/basecall/CudaCaller.cpp:            spdlog::warn("Unexpected model insize {}. Estimating GPU memory requirements.");
dorado/basecall/CudaCaller.cpp:    // See `dorado::basecall::decode::CUDADecoder::beam_search_part_1()` for more details.
dorado/basecall/CudaCaller.cpp:void CudaCaller::determine_batch_dims(const BasecallerCreationParams &params) {
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDAGuard device_guard(m_options.device());
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDACachingAllocator::emptyCache();
dorado/basecall/CudaCaller.cpp:    // If running on a Jetson device with unified memory for CPU and GPU we can't use all
dorado/basecall/CudaCaller.cpp:    // the available memory for GPU tasks. This way we leave at least half for the CPU,
dorado/basecall/CudaCaller.cpp:    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
dorado/basecall/CudaCaller.cpp:    int64_t gpu_mem_limit = int64_t(available * memory_limit_fraction - GB);
dorado/basecall/CudaCaller.cpp:    if (gpu_mem_limit < 0) {
dorado/basecall/CudaCaller.cpp:        spdlog::warn("Failed to determine safe batch size. Less than 1GB GPU memory available.");
dorado/basecall/CudaCaller.cpp:    spdlog::debug("{} memory limit {:.2f}GB", m_device, gpu_mem_limit / GB);
dorado/basecall/CudaCaller.cpp:    // user choice. This makes sure batch size is compatible with GPU kernels.
dorado/basecall/CudaCaller.cpp:        int max_batch_size = int(gpu_mem_limit / bytes_per_chunk);
dorado/basecall/CudaCaller.cpp:                    "minimum is {}, GPU may run out of memory.",
dorado/basecall/CudaCaller.cpp:    const auto &chunk_benchmarks = CudaChunkBenchmarks::instance().get_chunk_timings(
dorado/basecall/CudaCaller.cpp:        spdlog::info(std::string("Calculating optimized batch size for GPU \"") + prop->name +
dorado/basecall/CudaCaller.cpp:                using utils::handle_cuda_result;
dorado/basecall/CudaCaller.cpp:                cudaEvent_t start, stop;
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventCreate(&start));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventCreate(&stop));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventRecord(start));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventRecord(stop));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventSynchronize(stop));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventElapsedTime(&ms, start, stop));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventDestroy(start));
dorado/basecall/CudaCaller.cpp:                handle_cuda_result(cudaEventDestroy(stop));
dorado/basecall/CudaCaller.cpp:            // Clear the cache each time. Without this, intermittent cuda memory allocation errors
dorado/basecall/CudaCaller.cpp:            // are seen on windows laptop NVIDIA RTX A5500 Laptop GPU. See JIRA issue DOR-466
dorado/basecall/CudaCaller.cpp:            c10::cuda::CUDACachingAllocator::emptyCache();
dorado/basecall/CudaCaller.cpp:        CudaChunkBenchmarks::instance().add_chunk_timings(prop->name, m_config.model_path.string(),
dorado/basecall/CudaCaller.cpp:                "Adding chunk timings to internal cache for GPU {}, model {} ({} "
dorado/basecall/CudaCaller.cpp:void CudaCaller::start_threads() {
dorado/basecall/CudaCaller.cpp:    m_cuda_thread = std::thread([this] { cuda_thread_fn(); });
dorado/basecall/CudaCaller.cpp:void CudaCaller::cuda_thread_fn() {
dorado/basecall/CudaCaller.cpp:    utils::set_thread_name("cuda_caller");
dorado/basecall/CudaCaller.cpp:            "cuda_thread_fn_device_" + std::to_string(m_options.device().index());
dorado/basecall/CudaCaller.cpp:    const std::string gpu_lock_scope_str = "gpu_lock_" + std::to_string(m_options.device().index());
dorado/basecall/CudaCaller.cpp:    c10::cuda::CUDAStreamGuard stream_guard(m_stream);
dorado/basecall/CudaCaller.cpp:        nvtxRangePushA(gpu_lock_scope_str.c_str());
dorado/basecall/CudaCaller.cpp:        auto gpu_lock = dorado::utils::acquire_gpu_lock(m_options.device().index(), !m_low_latency);
dorado/basecall/CudaCaller.cpp:                c10::cuda::CUDACachingAllocator::getDeviceStats(m_options.device().index());
dorado/basecall/CudaCaller.cpp:        auto print_stat = [](c10::cuda::CUDACachingAllocator::StatArray &st) {
dorado/basecall/CudaCaller.cpp:            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.msg());
dorado/basecall/CudaCaller.cpp:            c10::cuda::CUDACachingAllocator::emptyCache();
dorado/basecall/nn/TxModel.cpp:#include "torch_utils/gpu_profiling.h"
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/TxModel.cpp:    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
dorado/basecall/nn/TxModel.cpp:    auto use_koi_swiglu = x.is_cuda() && utils::get_dev_opt<bool>("use_koi_swiglu", true) &&
dorado/basecall/nn/TxModel.cpp:        auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/TxModel.cpp:            x.is_cuda() && utils::get_dev_opt<bool>("use_koi_rote", true) && d_model <= 512;
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/TxModel.cpp:            auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
dorado/basecall/nn/TxModel.cpp:    bool use_koi_attention = x.is_cuda() && utils::get_dev_opt<bool>("use_koi_attention", true) &&
dorado/basecall/nn/TxModel.cpp:        auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
dorado/basecall/nn/TxModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/TxModel.cpp:        if (x.is_cuda()) {
dorado/basecall/nn/TxModel.cpp:            auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
dorado/basecall/nn/TxModel.cpp:#if DORADO_CUDA_BUILD && !defined(DORADO_TX2)
dorado/basecall/nn/MetalCRFModel.cpp:// Splitting up command buffers can be useful since it allows Xcode to make GPU captures.
dorado/basecall/nn/MetalCRFModel.cpp:// We assume non-AMD GPUs, in which case this is 32.
dorado/basecall/nn/TxModel.h:#include "torch_utils/gpu_profiling.h"
dorado/basecall/nn/CRFModel.cpp:#include "torch_utils/gpu_profiling.h"
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#include "torch_utils/cuda_utils.h"
dorado/basecall/nn/CRFModel.cpp:#include <ATen/cuda/CUDAContext.h>
dorado/basecall/nn/CRFModel.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:// TODO: Add Cutlass kernels for 7.0 (V100, FP16) and for GPUs with less shared memory (7.x, 8.x)
dorado/basecall/nn/CRFModel.cpp:    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
dorado/basecall/nn/CRFModel.cpp:    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
dorado/basecall/nn/CRFModel.cpp:    // This may be overly conservative, but all CUDA allocation functions are guaranteed to
dorado/basecall/nn/CRFModel.cpp:    // return 256-byte aligned pointers (even though GPU cache lines are at most 128 bytes).
dorado/basecall/nn/CRFModel.cpp:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/CRFModel.cpp:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/CRFModel.cpp:            // Move weights to GPU if called for the first time
dorado/basecall/nn/CRFModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/CRFModel.cpp:        // Move weights to GPU if called for the first time
dorado/basecall/nn/CRFModel.cpp:    // Quantise weights and move to GPU, if called for the first time
dorado/basecall/nn/CRFModel.cpp:    auto stream = at::cuda::getCurrentCUDAStream().stream();
dorado/basecall/nn/CRFModel.cpp:        dorado::utils::handle_cuda_result(host_small_lstm(
dorado/basecall/nn/CRFModel.cpp:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:    c10::cuda::CUDAGuard device_guard(in.device());
dorado/basecall/nn/CRFModel.cpp:    // `CUDADecoder` on reading the scores. This eliminates the cost of a large matrix
dorado/basecall/nn/CRFModel.cpp:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.cpp:    if (x.is_cuda() && x.dtype() == torch::kF16) {
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#endif  // if DORADO_CUDA_BUILD
dorado/basecall/nn/CRFModel.h:#if DORADO_CUDA_BUILD
dorado/basecall/CMakeLists.txt:if (DORADO_GPU_BUILD)
dorado/basecall/CMakeLists.txt:            CudaChunkBenchmarks.cpp
dorado/basecall/CMakeLists.txt:            CudaChunkBenchmarks.h
dorado/basecall/CMakeLists.txt:            CudaModelRunner.cpp
dorado/basecall/CMakeLists.txt:            CudaModelRunner.h
dorado/basecall/CMakeLists.txt:            CudaCaller.cpp
dorado/basecall/CMakeLists.txt:            CudaCaller.h
dorado/basecall/CMakeLists.txt:            decode/CUDADecoder.cpp
dorado/basecall/CMakeLists.txt:            decode/CUDADecoder.h
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_A100_80GB_PCIe.cpp
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_A100_80GB_PCIe.h
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_H100_PCIe.cpp
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_H100_PCIe.h
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_RTX_A6000.cpp
dorado/basecall/CMakeLists.txt:            benchmarks/NVIDIA_RTX_A6000.h
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.h:void AddNVIDIA_H100_PCIeBenchmarks(std::map<std::pair<std::string, std::string>,
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.h:void AddNVIDIA_RTX_A6000Benchmarks(std::map<std::pair<std::string, std::string>,
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:#include "NVIDIA_RTX_A6000.h"
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:void AddNVIDIA_RTX_A6000Benchmarks(std::map<std::pair<std::string, std::string>,
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_260bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_260bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_260bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_fast@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_fast@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r9.4.1_e8_fast@v3.4"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r9.4.1_e8_hac@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "dna_r9.4.1_e8_sup@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "rna004_130bps_fast@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "rna004_130bps_hac@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_RTX_A6000.cpp:    chunk_benchmarks[{"NVIDIA RTX A6000", "rna004_130bps_sup@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:#include "NVIDIA_H100_PCIe.h"
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:void AddNVIDIA_H100_PCIeBenchmarks(std::map<std::pair<std::string, std::string>,
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_260bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_260bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_260bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_fast@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_fast@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r9.4.1_e8_fast@v3.4"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r9.4.1_e8_hac@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "dna_r9.4.1_e8_sup@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "rna004_130bps_fast@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "rna004_130bps_hac@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_H100_PCIe.cpp:    chunk_benchmarks[{"NVIDIA H100 PCIe", "rna004_130bps_sup@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:#include "NVIDIA_A100_80GB_PCIe.h"
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:void AddNVIDIA_A100_80GB_PCIeBenchmarks(
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_260bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_260bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_260bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_fast@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_fast@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_fast@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_hac@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_hac@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_sup@v4.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_sup@v4.3.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r10.4.1_e8.2_400bps_sup@v5.0.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r9.4.1_e8_fast@v3.4"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r9.4.1_e8_hac@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "dna_r9.4.1_e8_sup@v3.3"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "rna004_130bps_fast@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "rna004_130bps_hac@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.cpp:    chunk_benchmarks[{"NVIDIA A100 80GB PCIe", "rna004_130bps_sup@v5.1.0"}] = {
dorado/basecall/benchmarks/NVIDIA_A100_80GB_PCIe.h:void AddNVIDIA_A100_80GB_PCIeBenchmarks(std::map<std::pair<std::string, std::string>,
dorado/basecall/MetalCaller.cpp:    // For unknown reasons, concurrent access to the GPU from multiple instances of this thread --
dorado/basecall/MetalCaller.cpp:        // allowing the GPU to proceed.
dorado/basecall/MetalCaller.cpp:        // Basecall the chunk and run the scan kernels on GPU
dorado/basecall/MetalCaller.cpp:                spdlog::critical("Failed to successfully submit GPU command buffers.");
dorado/basecall/MetalCaller.cpp:                throw std::runtime_error("Failed to successfully submit GPU command buffers.");
dorado/basecall/MetalCaller.cpp:                // Now that all chunks are decoded, signal that the GPU can overwrite the scores
dorado/basecall/MetalCaller.cpp:    // with neural network GPU buffers and CPU buffers assumed to occupy a subset of the
dorado/basecall/MetalCaller.cpp:    // the maximum GPU cores when running sup models on systems with a large GPU core
dorado/basecall/MetalCaller.cpp:    // that will use 1/4 of GPU cores for LSTM execution.
dorado/3rdparty/catch2/catch2/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
dorado/3rdparty/catch2/catch2/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
dorado/3rdparty/catch2/catch2/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2
dorado/read_pipeline/HtsWriter.h:    HtsWriter(utils::HtsFile& file, std::string gpu_names);
dorado/read_pipeline/HtsWriter.h:    std::string m_gpu_names{};
dorado/read_pipeline/CorrectionInferenceNode.h:    std::array<std::mutex, 32> m_gpu_mutexes;
dorado/read_pipeline/HtsWriter.cpp:HtsWriter::HtsWriter(utils::HtsFile& file, std::string gpu_names)
dorado/read_pipeline/HtsWriter.cpp:        : MessageSink(10000, 1), m_file(file), m_gpu_names(std::move(gpu_names)) {
dorado/read_pipeline/HtsWriter.cpp:    if (!m_gpu_names.empty()) {
dorado/read_pipeline/HtsWriter.cpp:        m_gpu_names = "gpu:" + m_gpu_names;
dorado/read_pipeline/HtsWriter.cpp:            if (!m_gpu_names.empty()) {
dorado/read_pipeline/HtsWriter.cpp:                bam_aux_append(aln.get(), "DS", 'Z', int(m_gpu_names.length() + 1),
dorado/read_pipeline/HtsWriter.cpp:                               (uint8_t*)m_gpu_names.c_str());
dorado/read_pipeline/ModBaseCallerNode.h:    // Worker threads, performs the GPU calls to the modbase models
dorado/read_pipeline/BasecallerNode.cpp:    // Model execution creates GPU-related autorelease objects.
dorado/read_pipeline/BasecallerNode.cpp:    // Allows optimal batch size to be used for every GPU
dorado/read_pipeline/BasecallerNode.cpp:        // [num_devices][num_gpu_runners][num_chunk_sizes] (see
dorado/read_pipeline/BasecallerNode.h:    // Vector of model runners (each with their own GPU access etc)
dorado/read_pipeline/CorrectionInferenceNode.cpp:#include "torch_utils/gpu_profiling.h"
dorado/read_pipeline/CorrectionInferenceNode.cpp:#if DORADO_CUDA_BUILD
dorado/read_pipeline/CorrectionInferenceNode.cpp:#include "torch_utils/cuda_utils.h"
dorado/read_pipeline/CorrectionInferenceNode.cpp:#if DORADO_CUDA_BUILD
dorado/read_pipeline/CorrectionInferenceNode.cpp:#include <c10/cuda/CUDACachingAllocator.h>
dorado/read_pipeline/CorrectionInferenceNode.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/read_pipeline/CorrectionInferenceNode.cpp:#if DORADO_CUDA_BUILD
dorado/read_pipeline/CorrectionInferenceNode.cpp:    if (device.is_cuda()) {
dorado/read_pipeline/CorrectionInferenceNode.cpp:        stream = c10::cuda::getStreamFromPool(false, device.index());
dorado/read_pipeline/CorrectionInferenceNode.cpp:    c10::cuda::OptionalCUDAStreamGuard guard(stream);
dorado/read_pipeline/CorrectionInferenceNode.cpp:        std::unique_lock<std::mutex> lock(m_gpu_mutexes[mtx_idx]);
dorado/read_pipeline/CorrectionInferenceNode.cpp:#if DORADO_CUDA_BUILD
dorado/read_pipeline/CorrectionInferenceNode.cpp:            spdlog::warn("Caught Torch error '{}', clearing CUDA cache and retrying.", e.what());
dorado/read_pipeline/CorrectionInferenceNode.cpp:            c10::cuda::CUDACachingAllocator::emptyCache();
dorado/read_pipeline/CorrectionInferenceNode.cpp:#if DORADO_CUDA_BUILD
dorado/read_pipeline/CorrectionInferenceNode.cpp:    else if (utils::starts_with(device, "cuda")) {
dorado/read_pipeline/CorrectionInferenceNode.cpp:        devices = dorado::utils::parse_cuda_device_string(device);
dorado/read_pipeline/CorrectionInferenceNode.cpp:            throw std::runtime_error("CUDA device requested but no devices found.");
dorado/read_pipeline/CorrectionInferenceNode.cpp:        const float batch_factor = (utils::starts_with(device, "cuda")) ? 0.4f : 0.8f;
dorado/correct/infer.cpp:#if DORADO_CUDA_BUILD
dorado/correct/infer.cpp:#include "torch_utils/cuda_utils.h"
dorado/correct/infer.cpp:#if DORADO_CUDA_BUILD
dorado/correct/infer.cpp:    else if (utils::starts_with(device, "cuda")) {
dorado/correct/infer.h:#include "torch_utils/gpu_profiling.h"
dorado/utils/CMakeLists.txt:        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
dorado/utils/CMakeLists.txt:        DORADO_CUDA_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<NOT:$<BOOL:${APPLE}>>>
dorado/utils/CMakeLists.txt:        DORADO_METAL_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<BOOL:${APPLE}>>
dorado/torch_utils/gpu_monitor.cpp:#include "gpu_monitor.h"
dorado/torch_utils/gpu_monitor.cpp:namespace dorado::utils::gpu_monitor {
dorado/torch_utils/gpu_monitor.cpp:                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml.dll",
dorado/torch_utils/gpu_monitor.cpp:                win64_dir + "\\NVIDIA Corporation\\NVSMI\\nvml\\lib\\nvml.dll",
dorado/torch_utils/gpu_monitor.cpp:                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml.dll",
dorado/torch_utils/gpu_monitor.cpp:                win64_dir + "\\NVIDIA Corporation\\GDK\\nvml\\lib\\nvml.dll",
dorado/torch_utils/gpu_monitor.cpp:        for (const char *path : {"libnvidia-ml.so.1", "libnvidia-ml.so"}) {
dorado/torch_utils/gpu_monitor.cpp:                          info.gpu_shutdown_temperature, info.gpu_shutdown_temperature_error);
dorado/torch_utils/gpu_monitor.cpp:                          info.gpu_slowdown_temperature, info.gpu_slowdown_temperature_error);
dorado/torch_utils/gpu_monitor.cpp:    assign_threshold_temp(nvml, device, NVML_TEMPERATURE_THRESHOLD_GPU_MAX,
dorado/torch_utils/gpu_monitor.cpp:                          info.gpu_max_operating_temperature,
dorado/torch_utils/gpu_monitor.cpp:                          info.gpu_max_operating_temperature_error);
dorado/torch_utils/gpu_monitor.cpp:    info.percentage_utilization_gpu = utilization.gpu;
dorado/torch_utils/gpu_monitor.cpp:    auto result = nvml->DeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value);
dorado/torch_utils/gpu_monitor.cpp:    std::ifstream version_file("/proc/driver/nvidia/version",
dorado/torch_utils/gpu_monitor.cpp:        spdlog::warn("No NVIDIA version file found in /proc");
dorado/torch_utils/gpu_monitor.cpp:        auto info = detail::parse_nvidia_version_line(line);
dorado/torch_utils/gpu_monitor.cpp:    auto info = detail::parse_nvidia_tegra_line(line);
dorado/torch_utils/gpu_monitor.cpp:std::optional<std::string> get_nvidia_driver_version() {
dorado/torch_utils/gpu_monitor.cpp:std::optional<std::string> parse_nvidia_version_line(std::string_view line) {
dorado/torch_utils/gpu_monitor.cpp:std::optional<std::string> parse_nvidia_tegra_line(const std::string &line) {
dorado/torch_utils/gpu_monitor.cpp:    // https://forums.developer.nvidia.com/t/how-do-i-know-what-version-of-l4t-my-jetson-tk1-is-running/38893
dorado/torch_utils/gpu_monitor.cpp:}  // namespace dorado::utils::gpu_monitor
dorado/torch_utils/cuda_utils.h:#include <cuda_runtime.h>
dorado/torch_utils/cuda_utils.h:// Returns a lock providing exclusive access to the GPU with the specified index.
dorado/torch_utils/cuda_utils.h:// attempting to allocate GPU memory on, or submit work to, the device in question.
dorado/torch_utils/cuda_utils.h:// the GPU is available to other users again.
dorado/torch_utils/cuda_utils.h:std::unique_lock<std::mutex> acquire_gpu_lock(int gpu_index, bool use_lock);
dorado/torch_utils/cuda_utils.h:// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of strings, one for
dorado/torch_utils/cuda_utils.h:// each device (e.g ["cuda:0", "cuda:2", ..., "cuda:7"]. This function will validate that the device IDs
dorado/torch_utils/cuda_utils.h:std::vector<std::string> parse_cuda_device_string(const std::string &device_string);
dorado/torch_utils/cuda_utils.h:// Try to parse the device string in the same manner parse_cuda_device_string
dorado/torch_utils/cuda_utils.h:bool try_parse_cuda_device_string(const std::string &device_string,
dorado/torch_utils/cuda_utils.h:struct CUDADeviceInfo {
dorado/torch_utils/cuda_utils.h:    cudaDeviceProp device_properties;
dorado/torch_utils/cuda_utils.h:// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a vector of CUDADeviceInfo for all
dorado/torch_utils/cuda_utils.h:std::vector<CUDADeviceInfo> get_cuda_device_info(const std::string &device_string,
dorado/torch_utils/cuda_utils.h:// Given a string representing cuda devices (e.g "cuda:0,1,3") returns a string containing
dorado/torch_utils/cuda_utils.h:// the set of types of gpu that will be used.
dorado/torch_utils/cuda_utils.h:std::string get_cuda_gpu_names(const std::string &device_string);
dorado/torch_utils/cuda_utils.h:// Print `label` and stats for Torch CUDACachingAllocator to stderr. Useful for tracking down
dorado/torch_utils/cuda_utils.h:// where Torch allocates GPU memory.
dorado/torch_utils/cuda_utils.h:void print_cuda_alloc_info(const std::string &label);
dorado/torch_utils/cuda_utils.h:// Deal with a result from a cudaGetLastError call.  May raise an exception to provide information to the user.
dorado/torch_utils/cuda_utils.h:void handle_cuda_result(int cuda_result);
dorado/torch_utils/torch_utils.cpp:#if DORADO_CUDA_BUILD
dorado/torch_utils/torch_utils.cpp:#include <c10/cuda/CUDAAllocatorConfig.h>
dorado/torch_utils/torch_utils.cpp:#include <c10/cuda/CUDACachingAllocator.h>
dorado/torch_utils/torch_utils.cpp:#endif  // DORADO_CUDA_BUILD
dorado/torch_utils/torch_utils.cpp:#if DORADO_CUDA_BUILD
dorado/torch_utils/torch_utils.cpp:#if DORADO_CUDA_BUILD && TORCH_VERSION_MAJOR >= 2
dorado/torch_utils/torch_utils.cpp:    const char *pytorch_cuda_alloc_conf = std::getenv("PYTORCH_CUDA_ALLOC_CONF");
dorado/torch_utils/torch_utils.cpp:    if (pytorch_cuda_alloc_conf != nullptr) {
dorado/torch_utils/torch_utils.cpp:        std::string_view str(pytorch_cuda_alloc_conf);
dorado/torch_utils/torch_utils.cpp:        settings += std::string(",") + pytorch_cuda_alloc_conf;
dorado/torch_utils/torch_utils.cpp:    c10::cuda::CUDACachingAllocator::setAllocatorSettings(settings);
dorado/torch_utils/CMakeLists.txt:    gpu_monitor.cpp
dorado/torch_utils/CMakeLists.txt:    gpu_monitor.h
dorado/torch_utils/CMakeLists.txt:    gpu_profiling.h
dorado/torch_utils/CMakeLists.txt:if (DORADO_GPU_BUILD)
dorado/torch_utils/CMakeLists.txt:            cuda_utils.cpp
dorado/torch_utils/CMakeLists.txt:            cuda_utils.h
dorado/torch_utils/CMakeLists.txt:        DORADO_GPU_BUILD=$<BOOL:${DORADO_GPU_BUILD}>
dorado/torch_utils/CMakeLists.txt:        DORADO_CUDA_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<NOT:$<BOOL:${APPLE}>>>
dorado/torch_utils/CMakeLists.txt:        DORADO_METAL_BUILD=$<AND:$<BOOL:${DORADO_GPU_BUILD}>,$<BOOL:${APPLE}>>
dorado/torch_utils/CMakeLists.txt:    if (DORADO_GPU_BUILD)
dorado/torch_utils/metal_utils.cpp:        spdlog::trace("Metal command buffer {}: {} GPU ms {} CPU ms succeeded (try {})", label,
dorado/torch_utils/metal_utils.cpp:                      1000.f * float(cb->GPUEndTime() - cb->GPUStartTime()),
dorado/torch_utils/metal_utils.cpp:    static int gpu_core_count = -1;
dorado/torch_utils/metal_utils.cpp:    if (gpu_core_count != -1) {
dorado/torch_utils/metal_utils.cpp:        return gpu_core_count;
dorado/torch_utils/metal_utils.cpp:    // Attempt to directly query the GPU core count.
dorado/torch_utils/metal_utils.cpp:    if (auto core_count_opt = retrieve_ioreg_prop("AGXAccelerator", "gpu-core-count");
dorado/torch_utils/metal_utils.cpp:        gpu_core_count = static_cast<int>(core_count_opt.value());
dorado/torch_utils/metal_utils.cpp:        spdlog::debug("Retrieved GPU core count of {} from IO Registry", gpu_core_count);
dorado/torch_utils/metal_utils.cpp:        return gpu_core_count;
dorado/torch_utils/metal_utils.cpp:    gpu_core_count = 8;
dorado/torch_utils/metal_utils.cpp:    spdlog::debug("Basing GPU core count on Metal device string {}", name);
dorado/torch_utils/metal_utils.cpp:        gpu_core_count = 16;
dorado/torch_utils/metal_utils.cpp:        gpu_core_count = 32;
dorado/torch_utils/metal_utils.cpp:        gpu_core_count = 64;
dorado/torch_utils/metal_utils.cpp:    } else if (name == "Apple M2 GPU" || name == "Apple M4 GPU") {
dorado/torch_utils/metal_utils.cpp:        // querying.  The M2 iPad Pro always has 10 GPU cores.  Note also that
dorado/torch_utils/metal_utils.cpp:        // "GPU" at the end.
dorado/torch_utils/metal_utils.cpp:        gpu_core_count = 10;
dorado/torch_utils/metal_utils.cpp:    spdlog::warn("Failed to retrieve GPU core count from IO Registry: using value of {}",
dorado/torch_utils/metal_utils.cpp:                 gpu_core_count);
dorado/torch_utils/metal_utils.cpp:    return gpu_core_count;
dorado/torch_utils/auto_detect_device.h:#if DORADO_CUDA_BUILD
dorado/torch_utils/auto_detect_device.h:#include <torch/cuda.h>
dorado/torch_utils/auto_detect_device.h:#elif DORADO_CUDA_BUILD
dorado/torch_utils/auto_detect_device.h:    return torch::cuda::is_available() ? "cuda:all" : "cpu";
dorado/torch_utils/cuda_utils.cpp:#include "cuda_utils.h"
dorado/torch_utils/cuda_utils.cpp:#include <ATen/cuda/CUDAContext.h>
dorado/torch_utils/cuda_utils.cpp:#include <c10/cuda/CUDAGuard.h>
dorado/torch_utils/cuda_utils.cpp:#include <cuda_runtime.h>
dorado/torch_utils/cuda_utils.cpp:#include <cuda_runtime_api.h>
dorado/torch_utils/cuda_utils.cpp:const std::string_view USAGE_HELP{"CUDA device string format: \"cuda:0,...,N\" or \"cuda:all\"."};
dorado/torch_utils/cuda_utils.cpp: * Wrapper around CUDA events to measure GPU timings.
dorado/torch_utils/cuda_utils.cpp:class CUDATimer {
dorado/torch_utils/cuda_utils.cpp:    cudaEvent_t m_start, m_stop;
dorado/torch_utils/cuda_utils.cpp:    CUDATimer(const CUDATimer &) = delete;
dorado/torch_utils/cuda_utils.cpp:    CUDATimer &operator=(const CUDATimer &) = delete;
dorado/torch_utils/cuda_utils.cpp:     * The timer will start once all previously submitted CUDA work
dorado/torch_utils/cuda_utils.cpp:    void start() { handle_cuda_result(cudaEventRecord(m_start)); }
dorado/torch_utils/cuda_utils.cpp:     * The timer will stop once all previously submitted CUDA work
dorado/torch_utils/cuda_utils.cpp:    void stop() { handle_cuda_result(cudaEventRecord(m_stop)); }
dorado/torch_utils/cuda_utils.cpp:     * Get the time spent on the GPU between the begin and end markers.
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventSynchronize(m_stop));
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventElapsedTime(&ms, m_start, m_stop));
dorado/torch_utils/cuda_utils.cpp:    CUDATimer() {
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventCreate(&m_start));
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventCreate(&m_stop));
dorado/torch_utils/cuda_utils.cpp:    ~CUDATimer() {
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventDestroy(m_start));
dorado/torch_utils/cuda_utils.cpp:        handle_cuda_result(cudaEventDestroy(m_stop));
dorado/torch_utils/cuda_utils.cpp:MatmulMode get_cuda_matmul_fp16_mode() {
dorado/torch_utils/cuda_utils.cpp:    cudaDeviceProp *prop = at::cuda::getCurrentDeviceProperties();
dorado/torch_utils/cuda_utils.cpp:        switch (get_cuda_matmul_fp16_mode()) {
dorado/torch_utils/cuda_utils.cpp:bool is_cuda_device_string(const std::string &device_string) {
dorado/torch_utils/cuda_utils.cpp:    return device_string.substr(0, 5) == "cuda:";
dorado/torch_utils/cuda_utils.cpp:    oss << "Invalid CUDA device index '" << device_id << "' from device string "
dorado/torch_utils/cuda_utils.cpp:        << std::quoted(device_string) << ", there are " << num_devices << " visible CUDA devices.";
dorado/torch_utils/cuda_utils.cpp:        error_message = "device string set to " + device_string + " but no CUDA devices available.";
dorado/torch_utils/cuda_utils.cpp:            error_message = "Invalid CUDA device string: " + device_string;
dorado/torch_utils/cuda_utils.cpp:        oss << "No device index found in CUDA device string " << std::quoted(device_string) << ".";
dorado/torch_utils/cuda_utils.cpp:bool try_parse_cuda_device_string(const std::string &device_string,
dorado/torch_utils/cuda_utils.cpp:    if (!details::try_parse_device_ids(device_string, torch::cuda::device_count(), device_ids,
dorado/torch_utils/cuda_utils.cpp:        devices.push_back("cuda:" + std::to_string(device_id));
dorado/torch_utils/cuda_utils.cpp:std::vector<std::string> parse_cuda_device_string(const std::string &device_string) {
dorado/torch_utils/cuda_utils.cpp:    if (!try_parse_cuda_device_string(device_string, devices, error_message)) {
dorado/torch_utils/cuda_utils.cpp:std::vector<CUDADeviceInfo> get_cuda_device_info(const std::string &device_string,
dorado/torch_utils/cuda_utils.cpp:    const auto num_devices = torch::cuda::device_count();
dorado/torch_utils/cuda_utils.cpp:    // Now inspect all the devices on the host to create the CUDADeviceInfo
dorado/torch_utils/cuda_utils.cpp:    std::vector<CUDADeviceInfo> results;
dorado/torch_utils/cuda_utils.cpp:        CUDADeviceInfo device_info;
dorado/torch_utils/cuda_utils.cpp:        cudaSetDevice(device_id);
dorado/torch_utils/cuda_utils.cpp:        cudaMemGetInfo(&device_info.free_mem, &device_info.total_mem);
dorado/torch_utils/cuda_utils.cpp:        cudaDeviceGetAttribute(&device_info.compute_cap_major, cudaDevAttrComputeCapabilityMajor,
dorado/torch_utils/cuda_utils.cpp:        cudaDeviceGetAttribute(&device_info.compute_cap_minor, cudaDevAttrComputeCapabilityMinor,
dorado/torch_utils/cuda_utils.cpp:        cudaGetDeviceProperties(&device_info.device_properties, device_id);
dorado/torch_utils/cuda_utils.cpp:            cudaDeviceReset();
dorado/torch_utils/cuda_utils.cpp:std::string get_cuda_gpu_names(const std::string &device_string) {
dorado/torch_utils/cuda_utils.cpp:    auto dev_info = utils::get_cuda_device_info(device_string, false);  // ignore unused GPUs
dorado/torch_utils/cuda_utils.cpp:    std::set<std::string> gpu_strs;
dorado/torch_utils/cuda_utils.cpp:    std::string gpu_names;
dorado/torch_utils/cuda_utils.cpp:        gpu_strs.insert(dev.device_properties.name);
dorado/torch_utils/cuda_utils.cpp:    for (const auto &gpu_id : gpu_strs) {
dorado/torch_utils/cuda_utils.cpp:        if (!gpu_names.empty()) {
dorado/torch_utils/cuda_utils.cpp:            gpu_names += "|";
dorado/torch_utils/cuda_utils.cpp:        gpu_names += gpu_id;
dorado/torch_utils/cuda_utils.cpp:    return gpu_names;
dorado/torch_utils/cuda_utils.cpp:std::unique_lock<std::mutex> acquire_gpu_lock(int gpu_index, bool use_lock) {
dorado/torch_utils/cuda_utils.cpp:    static std::vector<std::mutex> gpu_mutexes(torch::cuda::device_count());
dorado/torch_utils/cuda_utils.cpp:    return (use_lock ? std::unique_lock<std::mutex>(gpu_mutexes.at(gpu_index))
dorado/torch_utils/cuda_utils.cpp:void print_cuda_alloc_info(const std::string &label) {
dorado/torch_utils/cuda_utils.cpp:    auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(0);
dorado/torch_utils/cuda_utils.cpp:    auto print_stat_array = [](c10::cuda::CUDACachingAllocator::StatArray &stat,
dorado/torch_utils/cuda_utils.cpp:    std::cerr << "CUDAAlloc cpaf, " << label << " ";
dorado/torch_utils/cuda_utils.cpp:    c10::cuda::CUDAGuard device_guard(device);
dorado/torch_utils/cuda_utils.cpp:    cudaMemGetInfo(&free, &total);
dorado/torch_utils/cuda_utils.cpp:void handle_cuda_result(int cuda_result) {
dorado/torch_utils/cuda_utils.cpp:    if (cuda_result == cudaSuccess) {
dorado/torch_utils/cuda_utils.cpp:    if (cuda_result == cudaErrorNoKernelImageForDevice) {
dorado/torch_utils/cuda_utils.cpp:                std::string("Dorado cannot support the CUDA device being used,"
dorado/torch_utils/cuda_utils.cpp:        throw std::runtime_error(std::string("Cuda error: ") +
dorado/torch_utils/cuda_utils.cpp:                                 cudaGetErrorString(cudaError_t(cuda_result)));
dorado/torch_utils/cuda_utils.cpp:    if (!is_cuda_device_string(device_string)) {
dorado/torch_utils/cuda_utils.cpp:        // Not an error as there are valid non cuda device strings, e.g. "cpu".
dorado/torch_utils/cuda_utils.cpp:    if (device_string == "cuda:all" || device_string == "cuda:auto") {
dorado/torch_utils/cuda_utils.cpp:    auto res = cublasGemmEx(at::cuda::getCurrentCUDABlasHandle(), CUBLAS_OP_N, CUBLAS_OP_N,
dorado/torch_utils/cuda_utils.cpp:                            CUDA_R_16F, int(B.stride(0)), A.data_ptr(), CUDA_R_16F,
dorado/torch_utils/cuda_utils.cpp:                            int(A.stride(0)), &HALF_ZERO, C.data_ptr(), CUDA_R_16F,
dorado/torch_utils/cuda_utils.cpp:                            int(C.stride(0)), CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
dorado/torch_utils/gpu_profiling.h:// Set this to >0 to enable output of GPU profiling information to stderr
dorado/torch_utils/gpu_profiling.h:// or use `dorado [basecaller|duplex] ... --devopts cuda_profile_level=<X> ...`
dorado/torch_utils/gpu_profiling.h:#define CUDA_PROFILE_LEVEL_DEFAULT 0
dorado/torch_utils/gpu_profiling.h:#if DORADO_CUDA_BUILD
dorado/torch_utils/gpu_profiling.h:#include "cuda_utils.h"
dorado/torch_utils/gpu_profiling.h:#include <ATen/cuda/CUDAContext.h>
dorado/torch_utils/gpu_profiling.h:#include <cuda_runtime.h>
dorado/torch_utils/gpu_profiling.h:// If `detail_level <= CUDA_PROFILE_TO_CERR_LEVEL`, this times a range and prints it to stderr so
dorado/torch_utils/gpu_profiling.h:                       get_dev_opt<int>("cuda_profile_level", CUDA_PROFILE_LEVEL_DEFAULT)) {
dorado/torch_utils/gpu_profiling.h:            m_stream = at::cuda::getCurrentCUDAStream().stream();
dorado/torch_utils/gpu_profiling.h:            handle_cuda_result(cudaEventCreate(&m_start));
dorado/torch_utils/gpu_profiling.h:            handle_cuda_result(cudaEventRecord(m_start, m_stream));
dorado/torch_utils/gpu_profiling.h:        cudaEvent_t stop;
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventCreate(&stop));
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventRecord(stop, m_stream));
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventSynchronize(stop));
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventElapsedTime(&timeMs, m_start, stop));
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventDestroy(m_start));
dorado/torch_utils/gpu_profiling.h:        handle_cuda_result(cudaEventDestroy(stop));
dorado/torch_utils/gpu_profiling.h:    cudaStream_t m_stream;
dorado/torch_utils/gpu_profiling.h:    cudaEvent_t m_start;
dorado/torch_utils/gpu_monitor.h:namespace dorado::utils::gpu_monitor {
dorado/torch_utils/gpu_monitor.h: * Get the installed NVIDIA driver version.
dorado/torch_utils/gpu_monitor.h:std::optional<std::string> get_nvidia_driver_version();
dorado/torch_utils/gpu_monitor.h:    std::optional<unsigned int> gpu_shutdown_temperature;  // Temperature at which the GPU will
dorado/torch_utils/gpu_monitor.h:    std::string gpu_shutdown_temperature_error;
dorado/torch_utils/gpu_monitor.h:    std::optional<unsigned int> gpu_slowdown_temperature;  // Temperature at which the GPU will
dorado/torch_utils/gpu_monitor.h:    std::string gpu_slowdown_temperature_error;
dorado/torch_utils/gpu_monitor.h:    std::optional<unsigned int> gpu_max_operating_temperature;  // GPU Temperature at which the GPU
dorado/torch_utils/gpu_monitor.h:    std::string gpu_max_operating_temperature_error;
dorado/torch_utils/gpu_monitor.h:            percentage_utilization_gpu;  // Percent of time over the past sample period during which one or more kernels was executing on the GPU
dorado/torch_utils/gpu_monitor.h:            percentage_utilization_error;  // Shared error reason retrieving utilization info (gpu and memory)
dorado/torch_utils/gpu_monitor.h:std::optional<std::string> parse_nvidia_version_line(std::string_view line);
dorado/torch_utils/gpu_monitor.h:std::optional<std::string> parse_nvidia_tegra_line(const std::string& line);
dorado/torch_utils/gpu_monitor.h:}  // namespace dorado::utils::gpu_monitor
.gitlab-ci.yml:  CUDA: "11.8"
.gitlab-ci.yml:  WIN_CUDA_TOOLKIT: "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v${CUDA}"
.gitlab-ci.yml:    - nvidia-docker
.gitlab-ci.yml:    - nvidia-docker-tegra-gpu
.gitlab-ci.yml:    - linux-arm64-gpu
.gitlab-ci.yml:    - nvidia-docker
.gitlab-ci.yml:    - cuda-${CUDA}
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-no-deps-20.04-cuda-${CUDA}.0:1.0
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-22.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:    - export CUDA_TOOLKIT=$(cygpath -u $(cygpath -d "${WIN_CUDA_TOOLKIT}"))
.gitlab-ci.yml:    - export BUILD_OPTIONS="-A x64 -T cuda=${CUDA} -DCUDAToolkit_ROOT=${CUDA_TOOLKIT} -D WIN_THREADS=4 -DBUILD_KOI_FROM_SOURCE=ON -DGITLAB_CI_TOKEN=${CI_JOB_TOKEN}"
.gitlab-ci.yml:    - export CUDA_TOOLKIT=$(cygpath -u $(cygpath -d "${WIN_CUDA_TOOLKIT}"))
.gitlab-ci.yml:    - export BUILD_OPTIONS="-A x64 -T cuda=${CUDA} -DCUDAToolkit_ROOT=${CUDA_TOOLKIT} -D WIN_THREADS=4"
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:build_archive:linux:x86:cuda12:
.gitlab-ci.yml:  image: nvcr.io/nvidia/pytorch:24.06-py3
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-centos7-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:    - export CUDA_TOOLKIT=$(cygpath -u $(cygpath -d "${WIN_CUDA_TOOLKIT}"))
.gitlab-ci.yml:    - export BUILD_OPTIONS="-A x64 -T cuda=${CUDA} -DCUDAToolkit_ROOT=${CUDA_TOOLKIT} -DBUILD_KOI_FROM_SOURCE=ON -DGITLAB_CI_TOKEN=${CI_JOB_TOKEN}"
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:# Test that you can run dorado in a clean cuda 20.04 environment
.gitlab-ci.yml:test_archive:linux:x86:20.04_nvidia:
.gitlab-ci.yml:  image: nvidia/cuda:${CUDA}.0-devel-ubuntu20.04
.gitlab-ci.yml:  image: nvcr.io/nvidia/l4t-base:r32.4.3
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:  image: ${DORADO_DOCKER_ROOT}/dorado-deps-20.04-cuda-${CUDA}.0:1.1
.gitlab-ci.yml:    - nvidia-docker
.gitlab-ci.yml:    - ${GPU}
.gitlab-ci.yml:    - nvidia-smi
.gitlab-ci.yml:    - ./benchmark/generate_chunk_auto_batchsize_benchmarks.sh ./dist/bin/dorado cuda:0
.gitlab-ci.yml:       - GPU: "gpu-a100"
.gitlab-ci.yml:       - GPU: "gpu-v100"
.gitlab-ci.yml:       - GPU: "gpu-a6000"
.gitlab-ci.yml:          - "cuda:all"

```
