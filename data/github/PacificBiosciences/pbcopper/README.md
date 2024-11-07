# https://github.com/PacificBiosciences/pbcopper

```console
tests/meson.build:  # cuda
tests/meson.build:  'src/cuda/test_AsciiConversion.cpp',
tests/src/cuda/test_AsciiConversion.cpp:#include <pbcopper/cuda/AsciiConversion.h>
tests/src/cuda/test_AsciiConversion.cpp:TEST(Cuda_AsciiConversion, SingleBase)
tests/src/cuda/test_AsciiConversion.cpp:        const auto result = Cuda::AsciiToBitmaskContainer<64, 2>(arr, 1);
tests/src/cuda/test_AsciiConversion.cpp:        const auto result = Cuda::AsciiToBitmaskContainer<64, 2>(arr, 1);
tests/src/cuda/test_AsciiConversion.cpp:        const auto result = Cuda::AsciiToBitmaskContainer<64, 2>(arr, 1);
tests/src/cuda/test_AsciiConversion.cpp:        const auto result = Cuda::AsciiToBitmaskContainer<64, 2>(arr, 1);
tests/src/cuda/test_AsciiConversion.cpp:TEST(Cuda_AsciiConversion, Comprehensive)
tests/src/cuda/test_AsciiConversion.cpp:        // test CUDA ASCII → 2-bit conversion
tests/src/cuda/test_AsciiConversion.cpp:        const auto comp1 = Cuda::AsciiToBitmaskContainer<64, 2>(arr1, length);
tests/src/cuda/test_AsciiConversion.cpp:        // test CUDA 2-bit conversion → ASCII
tests/src/cuda/test_AsciiConversion.cpp:        Cuda::BitmaskContainerToAscii(comp1, length, arr2);
tests/src/cuda/test_AsciiConversion.cpp:        // test CUDA ASCII → 2-bit conversion (roundtrip)
tests/src/cuda/test_AsciiConversion.cpp:        const auto comp2 = Cuda::AsciiToBitmaskContainer<64, 2>(arr2, length);
include/meson.build:  # pbcopper/cuda
include/meson.build:      'pbcopper/cuda/AsciiConversion.h',
include/meson.build:      'pbcopper/cuda/Cuda.h',
include/meson.build:      'pbcopper/cuda/ThreadBlockHelper.h',
include/meson.build:    subdir : 'pbcopper/cuda')
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr static std::int32_t Capacity() { return CAPACITY_; }
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr explicit BitmaskContainer(
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr static BitmaskContainer MakeFromArray(
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr ValueType operator[](
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Clear() noexcept { data_ = 0; }
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Set(const std::int32_t idx,
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Remove(const std::int32_t idx) noexcept
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Insert(const std::int32_t idx,
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr static ComputationType GenerateMovePattern(
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void ReverseImpl() noexcept
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Reverse() noexcept
include/pbcopper/container/BitmaskContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr UnderlyingType RawData() const noexcept { return data_; }
include/pbcopper/container/BitContainer.h:/// tightly packed std::vector that is constexpr and GPU friendly. It is the
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr BitContainer(const UnderlyingType val,
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr std::int32_t Size() const noexcept { return size_; }
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr ValueType operator[](
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Clear() noexcept
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Set(const std::int32_t idx,
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Remove(const std::int32_t idx) noexcept
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Insert(const std::int32_t idx,
include/pbcopper/container/BitContainer.h:#ifndef __CUDA_ARCH__
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void PushBack(const ValueType val) noexcept
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void Reverse() noexcept
include/pbcopper/container/BitContainer.h:#ifndef __CUDA_ARCH__  // host
include/pbcopper/container/BitContainer.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr BitContainer Range(const std::int32_t pos,
include/pbcopper/container/BitContainer.h:#ifndef __CUDA_ARCH__  // host
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr std::int32_t SizeImpl() const noexcept
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr explicit DNA2bitStringImpl(const Base base) noexcept
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr explicit DNA2bitStringImpl(
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr std::int32_t Length() const noexcept
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr void ReverseComplement() noexcept
include/pbcopper/container/DNAString.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr DNA2bitStringImpl Range(
include/pbcopper/container/BitConversion.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr std::int32_t ConvertAsciiTo2bit(
include/pbcopper/container/BitConversion.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr char Convert2bitToAscii(const std::int32_t val) noexcept
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:        // TODO: remove once we're on C++20 in CUDA too
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:        // TODO: remove once we're on C++20 in CUDA too
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:// no __CUDA_ARCH__ code paths, since CUDA has no __builtin_ctz equivalent
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:        // TODO: remove once we're on C++20 in CUDA too
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:        // TODO: remove once we're on C++20 in CUDA too
include/pbcopper/utility/Intrinsics.h:#ifdef __CUDA_ARCH__
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr bool HasSingleBit(const T x) noexcept
include/pbcopper/utility/Intrinsics.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr int IntegralLog2(const T x)
include/pbcopper/utility/FastMod.h:#elif __CUDACC__
include/pbcopper/data/SNR.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr SNR(const float a, const float c, const float g,
include/pbcopper/data/SNR.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr SNR(const float (&snrs)[4]) noexcept
include/pbcopper/data/SNR.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr const float& operator[](
include/pbcopper/data/SNR.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr float& operator[](const std::int32_t i) noexcept
include/pbcopper/data/SNR.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr SNR ClampSNR(SNR v, const SNR& lo, const SNR& hi) noexcept
include/pbcopper/data/SNR.h:#ifndef __CUDA_ARCH__  // host
include/pbcopper/data/CigarOperation.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr static CigarOperationType CharToType(const char c)
include/pbcopper/data/CigarOperation.h:    PB_CUDA_HOST PB_CUDA_DEVICE
include/pbcopper/data/CigarOperation.h:    PB_CUDA_HOST PB_CUDA_DEVICE
include/pbcopper/data/CigarOperation.h:#ifndef __CUDA_ARCH__  // host
include/pbcopper/data/CigarOperation.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr std::uint32_t Length() const noexcept
include/pbcopper/data/CigarOperation.h:    PB_CUDA_HOST PB_CUDA_DEVICE constexpr CigarOperationType Type() const noexcept
include/pbcopper/PbcopperConfig.h:#ifdef __CUDACC__
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_HOST
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_HOST __host__
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_DEVICE
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_DEVICE __device__
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_CONSTANT
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_CONSTANT __constant__
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_FORCEINLINE
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_FORCEINLINE __forceinline__
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_HOST
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_HOST
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_DEVICE
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_DEVICE
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_CONSTANT
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_CONSTANT
include/pbcopper/PbcopperConfig.h:#ifndef PB_CUDA_FORCEINLINE
include/pbcopper/PbcopperConfig.h:#define PB_CUDA_FORCEINLINE
include/pbcopper/PbcopperConfig.h:constexpr unsigned int PB_CUDA_WARP_SIZE = 32U;
include/pbcopper/PbcopperConfig.h:constexpr unsigned int PB_CUDA_FULL_MASK = 0xFFFF'FFFFU;
include/pbcopper/numeric/Helper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr bool IsPowerOfTwo(const std::uint32_t x) noexcept
include/pbcopper/numeric/Helper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr T RoundUpDivision(const T x) noexcept
include/pbcopper/numeric/Helper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr T RoundDownDivision(const T x) noexcept
include/pbcopper/numeric/Helper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr T RoundUpToNextMultiple(const T x) noexcept
include/pbcopper/numeric/Helper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr T RoundDownToNextMultiple(const T x) noexcept
include/pbcopper/cuda/AsciiConversion.h:#ifndef PBCOPPER_CUDA_ASCIICONVERSION_H
include/pbcopper/cuda/AsciiConversion.h:#define PBCOPPER_CUDA_ASCIICONVERSION_H
include/pbcopper/cuda/AsciiConversion.h:namespace Cuda {
include/pbcopper/cuda/AsciiConversion.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr Container::BitmaskContainer<TotalBits, ElementBits>
include/pbcopper/cuda/AsciiConversion.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr void BitmaskContainerToAscii(
include/pbcopper/cuda/AsciiConversion.h:}  // namespace Cuda
include/pbcopper/cuda/AsciiConversion.h:#endif  // PBCOPPER_CUDA_ASCIICONVERSION_H
include/pbcopper/cuda/ThreadBlockHelper.h:#ifndef PBCOPPER_CUDA_THREADBLOCKHELPER_H
include/pbcopper/cuda/ThreadBlockHelper.h:#define PBCOPPER_CUDA_THREADBLOCKHELPER_H
include/pbcopper/cuda/ThreadBlockHelper.h:namespace Cuda {
include/pbcopper/cuda/ThreadBlockHelper.h:PB_CUDA_HOST PB_CUDA_DEVICE constexpr auto TotalThreadsBlock(const dim3 block) noexcept
include/pbcopper/cuda/ThreadBlockHelper.h:}  // namespace Cuda
include/pbcopper/cuda/ThreadBlockHelper.h:#endif  // PBCOPPER_CUDA_THREADBLOCKHELPER_H
include/pbcopper/cuda/Cuda.h:#ifndef PBCOPPER_CUDA_CUDA_H
include/pbcopper/cuda/Cuda.h:#define PBCOPPER_CUDA_CUDA_H
include/pbcopper/cuda/Cuda.h:#include <cuda_runtime_api.h>
include/pbcopper/cuda/Cuda.h:namespace Cuda {
include/pbcopper/cuda/Cuda.h:// Convenience allocators/deleters intended for use with GPU code
include/pbcopper/cuda/Cuda.h:            cudaHostAlloc(reinterpret_cast<void**>(&ptr), size, cudaHostAllocWriteCombined);
include/pbcopper/cuda/Cuda.h:        if (status != cudaSuccess) {
include/pbcopper/cuda/Cuda.h:        const auto status = cudaMallocHost(reinterpret_cast<void**>(&ptr), size);
include/pbcopper/cuda/Cuda.h:        if (status != cudaSuccess) {
include/pbcopper/cuda/Cuda.h:        // counter-part to cudaMallocHost on the CPU
include/pbcopper/cuda/Cuda.h:        cudaFreeHost(ptr);
include/pbcopper/cuda/Cuda.h:        const auto status = cudaMalloc(reinterpret_cast<void**>(&ptr), size);
include/pbcopper/cuda/Cuda.h:        if (status != cudaSuccess) {
include/pbcopper/cuda/Cuda.h:struct GPUMemoryDeleter
include/pbcopper/cuda/Cuda.h:        // counter-part to cudaMalloc on the GPU
include/pbcopper/cuda/Cuda.h:        cudaFree(ptr);
include/pbcopper/cuda/Cuda.h:}  // namespace Cuda
include/pbcopper/cuda/Cuda.h:#endif  // PBCOPPER_CUDA_CUDA_H
third-party/simde/x86/sse.h: *   2015      Brandon Rowlett <browlett@nvidia.com>
third-party/simde/x86/sse2.h: *   2015      Brandon Rowlett <browlett@nvidia.com>

```
