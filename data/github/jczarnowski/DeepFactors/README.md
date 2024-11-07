# https://github.com/jczarnowski/DeepFactors

```console
sources/gui/keyframe_renderer.cpp:  // TODO: check if a keyframe is currently on GPU, if yes then use that!
sources/core/deepfactors_options.h:  std::size_t     gpu = 0;
sources/core/deepfactors.cpp:#include "cuda_context.h"
sources/core/deepfactors.cpp:    int width = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).width();
sources/core/deepfactors.cpp:    int height = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).height();
sources/core/deepfactors.cpp:    vc::Image2DManaged<float, vc::TargetDeviceCUDA> img2gpu(width, height);
sources/core/deepfactors.cpp:      se3_aligner->Warp(relpose, cam_pyr[lvl], kf0->pyr_img.GetGpuLevel(lvl), kf1->pyr_img.GetGpuLevel(lvl),
sources/core/deepfactors.cpp:                        kf0->pyr_dpt.GetGpuLevel(lvl), img2gpu);
sources/core/deepfactors.cpp:      img2cpu.copyFrom(img2gpu);
sources/core/deepfactors.cpp:  // pop cuda context here so that tensorflow can clean up
sources/core/deepfactors.cpp:  cuda::PopContext();
sources/core/deepfactors.cpp:  // initialize the gpu
sources/core/deepfactors.cpp:  InitGpu(opts_.gpu);
sources/core/deepfactors.cpp:  // preprocess the image and upload it to gpu buffers
sources/core/deepfactors.cpp:  if (new_opts.gpu != opts_.gpu ||
sources/core/deepfactors.cpp:    LOG(FATAL) << "Online changes to GPU or network path are not allowed";
sources/core/deepfactors.cpp:void DeepFactors<Scalar,CS>::InitGpu(std::size_t device_id)
sources/core/deepfactors.cpp:  // explicitly create our own cuda context on selected gpu
sources/core/deepfactors.cpp:  // create a new context that we will use for our CUDA and bind to it
sources/core/deepfactors.cpp:  cuda::Init();
sources/core/deepfactors.cpp:  cuda::CreateAndBindContext(device_id);
sources/core/deepfactors.cpp:    auto scoped_pop = cuda::ScopedContextPop();
sources/core/network/tfwrap.h:  void gpu_memory_allow_growth(bool val)
sources/core/network/tfwrap.h:    gpu_memory_allow_growth_ = val;
sources/core/network/tfwrap.h:    TF_Buffer* buf = TF_CreateConfig(enable_xla_compilation_, gpu_memory_allow_growth_, num_cpu_devices_);
sources/core/network/tfwrap.h:  bool gpu_memory_allow_growth_ = false;
sources/core/network/decoder_network.cpp:  opts.gpu_memory_allow_growth(true);
sources/core/network/decoder_network.cpp:   * TODO TensorFlow doesn't allow us to set the visible device list to choose the GPU.
sources/core/network/decoder_network.cpp:   * By default, GPU 0 is selected so it works for us for now, but this absolutely must be fixed.
sources/core/system/loop_detector.h:  typedef vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> ImagePyramid;
sources/core/system/loop_detector.h:  typedef vc::RuntimeBufferPyramidManaged<ImageGrad, vc::TargetDeviceCUDA> GradPyramid;
sources/core/system/camera_tracker.cpp:                                        kf_->pyr_img.GetGpuLevel(level),
sources/core/system/camera_tracker.cpp:                                        kf_->pyr_dpt.GetGpuLevel(level),
sources/core/system/camera_tracker.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> warped(pyr_img1[level].width(), pyr_img1[level].height());
sources/core/system/camera_tracker.cpp:  se3aligner_.Warp(pose_ck_, camera_pyr_[level], kf_->pyr_img.GetGpuLevel(level), pyr_img1[level], kf_->pyr_dpt.GetGpuLevel(level), warped);
sources/core/system/camera_tracker.cpp:  kfimg_host.copyFrom(kf_->pyr_img.GetGpuLevel(level));
sources/core/system/camera_tracker.h:  typedef vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> ImageBufferPyramid;
sources/core/system/camera_tracker.h:  typedef vc::RuntimeBufferPyramidManaged<GradType, vc::TargetDeviceCUDA> GradBufferPyramid;
sources/core/CMakeLists.txt:  df_cuda
sources/core/mapping/keyframe.h:  // buffers that can exist on CPU or GPU
sources/core/mapping/mapper.cpp:#include "cuda_context.h"
sources/core/mapping/mapper.cpp:    int width = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).width();
sources/core/mapping/mapper.cpp:    int height = map->keyframes.Get(1)->pyr_img.GetGpuLevel(lvl).height();
sources/core/mapping/mapper.cpp:    vc::Image2DManaged<float, vc::TargetDeviceCUDA> img2gpu(width, height);
sources/core/mapping/mapper.cpp:      se3_aligner->Warp(relpose, cam_pyr[lvl], kf0->pyr_img.GetGpuLevel(lvl), kf1->pyr_img.GetGpuLevel(lvl),
sources/core/mapping/mapper.cpp:                        kf0->pyr_dpt.GetGpuLevel(lvl), img2gpu);
sources/core/mapping/mapper.cpp:      img2cpu.copyFrom(img2gpu);
sources/core/mapping/mapper.cpp:      df::UpdateDepth(cde, kf->pyr_prx_orig.GetGpuLevel(i),
sources/core/mapping/mapper.cpp:                         kf->pyr_jac.GetGpuLevel(i),
sources/core/mapping/mapper.cpp:                         kf->pyr_dpt.GetGpuLevel(i));
sources/core/mapping/mapper.cpp:    vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(i), 1.0f);
sources/core/mapping/mapper.cpp:      kf->pyr_img.GetGpuLevel(0).copyFrom(tmp1);
sources/core/mapping/mapper.cpp:      df::SobelGradients(kf->pyr_img.GetGpuLevel(0), kf->pyr_grad.GetGpuLevel(0));
sources/core/mapping/mapper.cpp:    df::GaussianBlurDown(kf->pyr_img.GetGpuLevel(i-1), kf->pyr_img.GetGpuLevel(i));
sources/core/mapping/mapper.cpp:    df::SobelGradients(kf->pyr_img.GetGpuLevel(i), kf->pyr_grad.GetGpuLevel(i));
sources/core/mapping/mapper.cpp:    cuda::ScopedContextPop pop;
sources/core/mapping/mapper.cpp:                       kf->pyr_prx_orig.GetGpuLevel(i),
sources/core/mapping/mapper.cpp:                       kf->pyr_jac.GetGpuLevel(i),
sources/core/mapping/mapper.cpp:                       kf->pyr_dpt.GetGpuLevel(i));
sources/core/mapping/mapper.cpp:    vc::Buffer2DManaged<typename Keyframe::GradT, vc::TargetDeviceCUDA> dpt_grad(w,h);
sources/core/mapping/mapper.cpp:    df::SobelGradients(kf->pyr_dpt.GetGpuLevel(0), dpt_grad);
sources/core/mapping/frame.h:        pyr_img.GetGpuLevel(0).copyFrom(tmp1);
sources/core/mapping/frame.h:        df::SobelGradients(pyr_img.GetGpuLevel(0), pyr_grad.GetGpuLevel(0));
sources/core/mapping/frame.h:      df::GaussianBlurDown(pyr_img.GetGpuLevel(i-1), pyr_img.GetGpuLevel(i));
sources/core/mapping/frame.h:      df::SobelGradients(pyr_img.GetGpuLevel(i), pyr_grad.GetGpuLevel(i));
sources/core/mapping/frame.h:  // buffers that can exist on CPU or GPU
sources/core/deepfactors.h:#include "cuda_context.h"
sources/core/deepfactors.h:  typedef vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> ImagePyrT;
sources/core/deepfactors.h:  typedef vc::RuntimeBufferPyramidManaged<Eigen::Matrix<Scalar,1,2>, vc::TargetDeviceCUDA> GradPyrT;
sources/core/deepfactors.h:  void InitGpu(std::size_t device_id);
sources/core/gtsam/photometric_factor.cpp:                                        kf_->pyr_img.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                        fr_->pyr_img.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                        kf_->pyr_dpt.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                        kf_->pyr_stdev.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                        fr_->pyr_grad.GetGpuLevel(i));
sources/core/gtsam/photometric_factor.cpp://  vc::Image2DView<Scalar,vc::TargetDeviceCUDA> vld;
sources/core/gtsam/photometric_factor.cpp://  vc::Image2DManaged<Scalar,vc::TargetDeviceCUDA> dummy_vld(cam_.width(), cam_.height());
sources/core/gtsam/photometric_factor.cpp://    vld = kf->pyr_vld.GetGpuLevel(i);
sources/core/gtsam/photometric_factor.cpp:  vc::Image2DView<Scalar,vc::TargetDeviceCUDA> vld = kf_->pyr_vld.GetGpuLevel(i);
sources/core/gtsam/photometric_factor.cpp:                                  kf_->pyr_img.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                  fr_->pyr_img.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                  kf_->pyr_dpt.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                  kf_->pyr_stdev.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                  kf_->pyr_jac.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                                  fr_->pyr_grad.GetGpuLevel(i));
sources/core/gtsam/photometric_factor.cpp:////      vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(0), 0.0f);
sources/core/gtsam/photometric_factor.cpp:  df::UpdateDepth(cde0, kf_->pyr_prx_orig.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                     kf_->pyr_jac.GetGpuLevel(i),
sources/core/gtsam/photometric_factor.cpp:                     kf_->pyr_dpt.GetGpuLevel(i));
sources/core/gtsam/depth_prior_factor.cpp:  // upload dpt image to GPU
sources/core/gtsam/depth_prior_factor.cpp:                                kf_->pyr_prx_orig.GetGpuLevel(i),
sources/core/gtsam/depth_prior_factor.cpp:                                kf_->pyr_jac.GetGpuLevel(i));
sources/core/gtsam/depth_prior_factor.cpp:                       kf_->pyr_prx_orig.GetGpuLevel(i),
sources/core/gtsam/depth_prior_factor.cpp:                       kf_->pyr_jac.GetGpuLevel(i),
sources/core/gtsam/depth_prior_factor.cpp:                       kf_->pyr_dpt.GetGpuLevel(i));
sources/core/gtsam/depth_prior_factor.h:  typedef vc::Buffer2DManaged<Scalar, vc::TargetDeviceCUDA> DepthBufferDevice;
sources/core/gtsam/depth_prior_factor.h:  vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> pyr_tgtdpt_;
sources/tools/CMakeLists.txt:# Benchmarking threads/blocks for key CUDA kernels
sources/tools/CMakeLists.txt:  df_cuda
sources/tools/kernel_benchmark.cpp:#include "cuda_context.h"
sources/tools/kernel_benchmark.cpp:DEFINE_uint32(gpu, 0, "Which gpu to use for SLAM");
sources/tools/kernel_benchmark.cpp:    vc::image::fillBuffer(kf->pyr_vld.GetGpuLevel(i), 1.0f);
sources/tools/kernel_benchmark.cpp:      kf->pyr_img.GetGpuLevel(0).copyFrom(tmp1);
sources/tools/kernel_benchmark.cpp:      df::SobelGradients(kf->pyr_img.GetGpuLevel(0), kf->pyr_grad.GetGpuLevel(0));
sources/tools/kernel_benchmark.cpp:    df::GaussianBlurDown(kf->pyr_img.GetGpuLevel(i-1), kf->pyr_img.GetGpuLevel(i));
sources/tools/kernel_benchmark.cpp:    df::SobelGradients(kf->pyr_img.GetGpuLevel(i), kf->pyr_grad.GetGpuLevel(i));
sources/tools/kernel_benchmark.cpp:    cuda::ScopedContextPop pop;
sources/tools/kernel_benchmark.cpp:  // init cuda context
sources/tools/kernel_benchmark.cpp:  auto devinfo = cuda::Init(FLAGS_gpu);
sources/tools/kernel_benchmark.cpp:    cuda::ScopedContextPop ctx;
sources/tools/kernel_benchmark.cpp:  // gpu warmup
sources/tools/kernel_benchmark.cpp:  vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
sources/tools/kernel_benchmark.cpp:  aligner->RunStep(pose0, pose1, zero_code, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                   kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                   kf0->pyr_stdev.GetGpuLevel(i), vld, kf0->pyr_jac.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                   kf1->pyr_grad.GetGpuLevel(i));
sources/tools/kernel_benchmark.cpp:      vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
sources/tools/kernel_benchmark.cpp:      aligner->RunStep(pose0, pose1, zero_code, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                       kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                       kf0->pyr_stdev.GetGpuLevel(i), vld, kf0->pyr_jac.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                       kf1->pyr_grad.GetGpuLevel(i));
sources/tools/kernel_benchmark.cpp:      vc::Image2DView<float,vc::TargetDeviceCUDA> vld = kf0->pyr_vld.GetGpuLevel(i);
sources/tools/kernel_benchmark.cpp:      aligner->EvaluateError(pose0, pose1, cam_pyr[i], kf0->pyr_img.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                             kf1->pyr_img.GetGpuLevel(i), kf0->pyr_dpt.GetGpuLevel(i),
sources/tools/kernel_benchmark.cpp:                             kf0->pyr_stdev.GetGpuLevel(i), kf1->pyr_grad.GetGpuLevel(i));
sources/common/algorithm/dense_sfm.h:          typename Device=vc::TargetDeviceCUDA,
sources/common/algorithm/dense_sfm.h:          typename Device=vc::TargetDeviceCUDA,
sources/CMakeLists.txt:add_subdirectory(cuda)
sources/demo/main.cpp:DEFINE_uint32(gpu, 0, "Which gpu to use for SLAM");
sources/demo/main.cpp:  opts.df_opts.gpu = FLAGS_gpu;
sources/cuda/cuda_context.h:#ifndef CUDA_CONTEXT_H_
sources/cuda/cuda_context.h:#define CUDA_CONTEXT_H_
sources/cuda/cuda_context.h:#include <cuda.h> // driver api
sources/cuda/cuda_context.h:namespace cuda
sources/cuda/cuda_context.h:DeviceInfo Init(uint gpuId = 0);
sources/cuda/cuda_context.h:DeviceInfo GetDeviceInfo(uint gpuId);
sources/cuda/cuda_context.h:#endif // CUDA_CONTEXT_H_
sources/cuda/cu_image_proc.cpp:__global__ void kernel_sobel_gradients(const vc::Buffer2DView<PixelT,vc::TargetDeviceCUDA> img,
sources/cuda/cu_image_proc.cpp:                                       vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA> grad)
sources/cuda/cu_image_proc.cpp:void SobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
sources/cuda/cu_image_proc.cpp:                    vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA>& grad)
sources/cuda/cu_image_proc.cpp:  // get a gaussian kernel in the gpu
sources/cuda/cu_image_proc.cpp:  cudaMemcpyToSymbol(SC, &coeffs, sizeof(SobelCoeffs));
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("cudaMemcpyToSymbol failed");
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("Kernel launch failed (kernel_sobel_gradients)");
sources/cuda/cu_image_proc.cpp:__global__ void kernel_gaussian_blur_down(const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> in,
sources/cuda/cu_image_proc.cpp:                                          vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> out)
sources/cuda/cu_image_proc.cpp:void GaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& in,
sources/cuda/cu_image_proc.cpp:                      vc::Buffer2DView<T, vc::TargetDeviceCUDA>& out)
sources/cuda/cu_image_proc.cpp:  // get a gaussian kernel in the gpu
sources/cuda/cu_image_proc.cpp:  cudaMemcpyToSymbol(gauss_coeffs, &coeffs, sizeof(coeffs));
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("cudaMemcpyToSymbol failed");
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("Kernel launch failed (kernel_gaussian_blur_down)");
sources/cuda/cu_image_proc.cpp:__global__ void kernel_squared_error(const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> buf1,
sources/cuda/cu_image_proc.cpp:                                     const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> buf2,
sources/cuda/cu_image_proc.cpp:                                     vc::Buffer1DView<Scalar, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_image_proc.cpp:T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
sources/cuda/cu_image_proc.cpp:               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2,
sources/cuda/cu_image_proc.cpp:               vc::Buffer1DView<T, vc::TargetDeviceCUDA>& bscratch)
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("[SquaredError] kernel launch failed");
sources/cuda/cu_image_proc.cpp:T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
sources/cuda/cu_image_proc.cpp:               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2)
sources/cuda/cu_image_proc.cpp:  vc::Buffer1DManaged<T, vc::TargetDeviceCUDA> bscratch(1024);
sources/cuda/cu_image_proc.cpp:                                    const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> prx_orig,
sources/cuda/cu_image_proc.cpp:                                    const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> prx_jac,
sources/cuda/cu_image_proc.cpp:                                    vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> dpt_out)
sources/cuda/cu_image_proc.cpp:  CudaCheckLastError("[UpdateDepth] kernel launch failed");
sources/cuda/cu_image_proc.cpp:template void GaussianBlurDown(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& in,
sources/cuda/cu_image_proc.cpp:                               vc::Buffer2DView<float, vc::TargetDeviceCUDA>& out);
sources/cuda/cu_image_proc.cpp://template void GaussianBlurDown(const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& in,
sources/cuda/cu_image_proc.cpp://                               vc::Buffer2DView<double, vc::TargetDeviceCUDA>& out);
sources/cuda/cu_image_proc.cpp:template void SobelGradients(const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& img,
sources/cuda/cu_image_proc.cpp:                             vc::Buffer2DView<Eigen::Matrix<float,1,2>,vc::TargetDeviceCUDA>& grad);
sources/cuda/cu_image_proc.cpp://template void SobelGradients(const vc::Buffer2DView<double,vc::TargetDeviceCUDA>& img,
sources/cuda/cu_image_proc.cpp://                             vc::Buffer2DView<Eigen::Matrix<double,1,2>,vc::TargetDeviceCUDA>& grad);
sources/cuda/cu_image_proc.cpp:template float SquaredError(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf1,
sources/cuda/cu_image_proc.cpp:                            const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& buf2);
sources/cuda/cu_image_proc.cpp://template double SquaredError(const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& buf1,
sources/cuda/cu_image_proc.cpp://                             const vc::Buffer2DView<double, vc::TargetDeviceCUDA>& buf2);
sources/cuda/cu_image_proc.cpp:                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
sources/cuda/cu_image_proc.cpp:                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
sources/cuda/cu_image_proc.cpp:                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);
sources/cuda/cu_image_proc.cpp://                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
sources/cuda/cu_image_proc.cpp://                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
sources/cuda/cu_image_proc.cpp://                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);
sources/cuda/cu_image_proc.cpp://                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_orig,
sources/cuda/cu_image_proc.cpp://                          const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& prx_jac,
sources/cuda/cu_image_proc.cpp://                          vc::Buffer2DView<float,vc::TargetDeviceCUDA>& dpt_out);
sources/cuda/cu_se3aligner.cpp:                                      vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_se3aligner.cpp:                                      vc::Buffer1DView<typename BaseT::CorrespondenceItem, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_se3aligner.cpp:  CudaCheckLastError("[SE3Aligner::Warp] Kernel launch failed (kernel_warp_calculate)");
sources/cuda/cu_se3aligner.cpp:  CudaCheckLastError("[SE3Aligner::Warp] Kernel launch failed (kernel_finalize_reduction)");
sources/cuda/cu_se3aligner.cpp:  CudaCheckLastError("[SE3Aligner::RunStep] Kernel launch failed");
sources/cuda/cu_sfmaligner.cpp:#include "cuda_context.h"
sources/cuda/cu_sfmaligner.cpp:    vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_sfmaligner.cpp:    vc::Buffer1DView<typename BaseT::ErrorReductionItem, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_sfmaligner.cpp:  cudaMemcpyToSymbol(sfm_params, &params.sfmparams, sizeof(DenseSfmParams));
sources/cuda/cu_sfmaligner.cpp:  CudaCheckLastError("Copying sfm parameters to gpu failed");
sources/cuda/cu_sfmaligner.cpp:  devinfo_ = cuda::GetCurrentDeviceInfo();
sources/cuda/cu_sfmaligner.cpp:  CudaCheckLastError("[SfmAligner::EvaluateError] kernel launch failed");
sources/cuda/cu_sfmaligner.cpp:  CudaCheckLastError("[SfmAligner::RunStep] kernel launch failed");
sources/cuda/synced_pyramid.h:  typedef vc::RuntimeBufferPyramidManaged<T, vc::TargetDeviceCUDA> GpuBuffer;
sources/cuda/synced_pyramid.h:      : gpu_modified_(false),
sources/cuda/synced_pyramid.h:     gpu_modified_ = other.gpu_modified_;
sources/cuda/synced_pyramid.h:     if (other.gpu_buf_)
sources/cuda/synced_pyramid.h:       gpu_buf_ = std::make_shared<GpuBuffer>(pyramid_levels_, width_, height_);
sources/cuda/synced_pyramid.h:       gpu_buf_->copyFrom(*other.gpu_buf_);
sources/cuda/synced_pyramid.h:  std::shared_ptr<const GpuBuffer> GetGpu() const
sources/cuda/synced_pyramid.h:    VLOG(LOG_LEVEL_COPY) << "Requesting const GPU pointer";
sources/cuda/synced_pyramid.h:    SynchronizeGpu();
sources/cuda/synced_pyramid.h:    return gpu_buf_;
sources/cuda/synced_pyramid.h:  std::shared_ptr<GpuBuffer> GetGpuMutable()
sources/cuda/synced_pyramid.h:    VLOG(LOG_LEVEL_COPY) << "Requesting mutable GPU pointer";
sources/cuda/synced_pyramid.h:    SynchronizeGpu();
sources/cuda/synced_pyramid.h:    FlagGpuNewData();
sources/cuda/synced_pyramid.h:    return gpu_buf_;
sources/cuda/synced_pyramid.h:  const typename GpuBuffer::ViewType& GetGpuLevel(int lvl) const
sources/cuda/synced_pyramid.h:    return GetGpu()->operator[](lvl);
sources/cuda/synced_pyramid.h:  typename GpuBuffer::ViewType& GetGpuLevel(int lvl)
sources/cuda/synced_pyramid.h:    return GetGpuMutable()->operator[](lvl);
sources/cuda/synced_pyramid.h:  void FlagGpuNewData()
sources/cuda/synced_pyramid.h:    gpu_modified_ = true;
sources/cuda/synced_pyramid.h:    VLOG(LOG_LEVEL_COPY) << "GPU buffer has been modified";
sources/cuda/synced_pyramid.h:  bool IsSynchronized() { return !gpu_modified_ && !cpu_modified_; }
sources/cuda/synced_pyramid.h:  void UnloadGpu() { gpu_buf_.reset(); FlagGpuNewData(); }
sources/cuda/synced_pyramid.h:    if (gpu_modified_ && cpu_modified_)
sources/cuda/synced_pyramid.h:      LOG(FATAL) << "Doth CPU and GPU data modified! Not sure which is valid";
sources/cuda/synced_pyramid.h:    // if there is new data from cpu, copy it to gpu
sources/cuda/synced_pyramid.h:    if (gpu_modified_)
sources/cuda/synced_pyramid.h:      VLOG(LOG_LEVEL_COPY) << "Copying buffer from GPU to CPU";
sources/cuda/synced_pyramid.h:      cpu_buf_->copyFrom(*gpu_buf_);
sources/cuda/synced_pyramid.h:    // we've synchronized so no new data on gpu
sources/cuda/synced_pyramid.h:    gpu_modified_ = false;
sources/cuda/synced_pyramid.h:  void SynchronizeGpu() const
sources/cuda/synced_pyramid.h:    if (!gpu_buf_)
sources/cuda/synced_pyramid.h:      gpu_buf_ = std::make_shared<GpuBuffer>(pyramid_levels_, width_, height_);
sources/cuda/synced_pyramid.h:      VLOG(LOG_LEVEL_COPY) << "Allocating a GPU buffer";
sources/cuda/synced_pyramid.h:    // if there is new data from cpu, copy it to gpu
sources/cuda/synced_pyramid.h:      VLOG(LOG_LEVEL_COPY) << "Copying buffer from CPU to GPU";
sources/cuda/synced_pyramid.h:      gpu_buf_->copyFrom(*cpu_buf_);
sources/cuda/synced_pyramid.h:  mutable std::shared_ptr<GpuBuffer> gpu_buf_;
sources/cuda/synced_pyramid.h:  mutable bool gpu_modified_;   // gpu nas new data
sources/cuda/device_info.h:namespace cuda
sources/cuda/device_info.h:} // namespace cuda
sources/cuda/cu_se3aligner.h:#include <VisionCore/CUDAGenerics.hpp>
sources/cuda/cu_se3aligner.h:  typedef vc::Image2DView<Scalar, vc::TargetDeviceCUDA> ImageBuffer;
sources/cuda/cu_se3aligner.h:  typedef vc::Image2DView<ImageGrad, vc::TargetDeviceCUDA> GradBuffer;
sources/cuda/cu_se3aligner.h:  typedef vc::Buffer1DManaged<CorrespondenceItem, vc::TargetDeviceCUDA> CorrespondenceReductionBuffer;
sources/cuda/cu_se3aligner.h:  typedef vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> StepReductionBuffer;
sources/cuda/kernel_utils.h:#include <VisionCore/CUDAGenerics.hpp>
sources/cuda/kernel_utils.h:#ifdef __CUDACC__
sources/cuda/kernel_utils.h:__global__ void kernel_finalize_reduction(const vc::Buffer1DView<T, vc::TargetDeviceCUDA> in,
sources/cuda/kernel_utils.h:                                          vc::Buffer1DView<T, vc::TargetDeviceCUDA> out,
sources/cuda/cuda_context.cpp:#include "cuda_context.h"
sources/cuda/cuda_context.cpp:#include <cuda_runtime.h> // runtime api
sources/cuda/cuda_context.cpp:#include <VisionCore/CUDAException.hpp>
sources/cuda/cuda_context.cpp:namespace cuda
sources/cuda/cuda_context.cpp:inline void ThrowOnErrorRuntime(const cudaError_t& err, const std::string& msg)
sources/cuda/cuda_context.cpp:  if (err != cudaSuccess)
sources/cuda/cuda_context.cpp:    throw vc::CUDAException(err, msg);
sources/cuda/cuda_context.cpp:  if (res != CUDA_SUCCESS)
sources/cuda/cuda_context.cpp:DeviceInfo GetDeviceInfo(uint gpuId)
sources/cuda/cuda_context.cpp:  // probe nvidia for device info
sources/cuda/cuda_context.cpp:  cudaDeviceProp cdp;
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaGetDeviceProperties(&cdp, gpuId), "cudaGetDeviceProperties failed");
sources/cuda/cuda_context.cpp:  // probe nvidia for free and total memory
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaMemGetInfo(&devInfo.FreeGlobalMem, &devInfo.TotalGlobalMem),
sources/cuda/cuda_context.cpp:                      "cudaMemGetInfo failed");
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaGetDevice(&dev), "cudaGetDevice failed");
sources/cuda/cuda_context.cpp:DeviceInfo Init(uint gpuId)
sources/cuda/cuda_context.cpp:  // select gpu
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaSetDevice(gpuId), "cudaSetDevice failed");
sources/cuda/cuda_context.cpp:  DeviceInfo devInfo = GetDeviceInfo(gpuId);
sources/cuda/cuda_context.cpp:  LOG(INFO) << "Selected GPU " << gpuId << ": " << devInfo.Name;
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaSetDevice(device_id), "cudaSetDevice failed");
sources/cuda/cuda_context.cpp:  ThrowOnErrorRuntime(cudaFree(0), "cudaFree failed");
sources/cuda/cuda_context.cpp:} // namespace cuda
sources/cuda/reduction_items.h:#include <VisionCore/CUDAGenerics.hpp>
sources/cuda/reduction_items.h:  #ifdef __CUDACC__
sources/cuda/reduction_items.h:  #ifdef __CUDACC__
sources/cuda/reduction_items.h:#ifdef __CUDACC__
sources/cuda/reduction_items.h:#endif // __CUDACC__
sources/cuda/cu_image_proc.h:void GaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& in,
sources/cuda/cu_image_proc.h:                      vc::Buffer2DView<T, vc::TargetDeviceCUDA>& out);
sources/cuda/cu_image_proc.h:void SobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
sources/cuda/cu_image_proc.h:                    vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA>& grad);
sources/cuda/cu_image_proc.h:T SquaredError(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf1,
sources/cuda/cu_image_proc.h:               const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& buf2);
sources/cuda/cu_image_proc.h:template <typename T, int CS, typename ImageBuf=vc::Buffer2DView<T, vc::TargetDeviceCUDA>>
sources/cuda/CMakeLists.txt:set(DF_CUDA_ARCH Auto CACHE STRING "A list of CUDA architectures to compile for. Specifying 'Auto' will attempt to autodetect available GPU devices")
sources/cuda/CMakeLists.txt:CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS ${DF_CUDA_ARCH})
sources/cuda/CMakeLists.txt:set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math)
sources/cuda/CMakeLists.txt:message("Compiling for CUDA architectures: ${CUDA_NVCC_ARCH_FLAGS}")
sources/cuda/CMakeLists.txt:list(APPEND CMAKE_CUDA_FLAGS -Xcudafe;--diag_suppress=esa_on_defaulted_function_ignored)
sources/cuda/CMakeLists.txt:  #list(APPEND CMAKE_CUDA_FLAGS --device-debug;--debug;-Xcompiler;-rdynamic;)#--ptxas-options=-v)
sources/cuda/CMakeLists.txt:  list(APPEND CMAKE_CUDA_FLAGS -g;-lineinfo)
sources/cuda/CMakeLists.txt:string(REPLACE ";" " " _TMP_STR "${CMAKE_CUDA_FLAGS}")
sources/cuda/CMakeLists.txt:set(CMAKE_CUDA_FLAGS "${_TMP_STR}")
sources/cuda/CMakeLists.txt:set(cuda_sources
sources/cuda/CMakeLists.txt:  cuda_context.cpp
sources/cuda/CMakeLists.txt:set(cuda_headers
sources/cuda/CMakeLists.txt:  cuda_context.h
sources/cuda/CMakeLists.txt:set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)
sources/cuda/CMakeLists.txt:add_library(df_cuda SHARED ${cuda_sources} ${cuda_headers})
sources/cuda/CMakeLists.txt:target_include_directories(df_cuda PUBLIC
sources/cuda/CMakeLists.txt:target_link_libraries(df_cuda PUBLIC
sources/cuda/CMakeLists.txt:  cuda
sources/cuda/CMakeLists.txt:#target_compile_features(df_cuda PUBLIC cxx_std_11)
sources/cuda/CMakeLists.txt:set_property(TARGET df_cuda PROPERTY CUDA_STANDARD 11)
sources/cuda/CMakeLists.txt:set_target_properties(df_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
sources/cuda/CMakeLists.txt:set_target_properties(df_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
sources/cuda/CMakeLists.txt:set_target_properties(df_cuda PROPERTIES RELOCATABLE_DEVICE_CODE ON)
sources/cuda/launch_utils.h:#include <VisionCore/CUDAException.hpp>
sources/cuda/launch_utils.h:inline void CudaCheckLastError(const std::string& msg)
sources/cuda/launch_utils.h:  cudaDeviceSynchronize();
sources/cuda/launch_utils.h:  cudaError_t err = cudaPeekAtLastError();
sources/cuda/launch_utils.h:  if (cudaSuccess != err)
sources/cuda/launch_utils.h:    throw vc::CUDAException(err, msg);
sources/cuda/cu_depthaligner.h:  typedef vc::Image2DView<Scalar,vc::TargetDeviceCUDA> ImageBuffer;
sources/cuda/cu_depthaligner.h:  vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch_;
sources/cuda/cu_depthaligner.cpp:                                  vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
sources/cuda/cu_depthaligner.cpp:  CudaCheckLastError("[DepthAligner::RunStep] kernel launch failed");
sources/cuda/cu_sfmaligner.h:#include <VisionCore/CUDAGenerics.hpp>
sources/cuda/cu_sfmaligner.h:  typedef vc::Image2DView<Scalar,vc::TargetDeviceCUDA> ImageBuffer;
sources/cuda/cu_sfmaligner.h:  typedef vc::Image2DView<ImageGrad,vc::TargetDeviceCUDA> GradBuffer;
sources/cuda/cu_sfmaligner.h:  cuda::DeviceInfo devinfo_;
sources/cuda/cu_sfmaligner.h:  vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch_;
sources/cuda/cu_sfmaligner.h:  vc::Buffer1DManaged<ErrorReductionItem, vc::TargetDeviceCUDA> bscratch2_;
thirdparty/makedeps.sh:              "-DUSE_OPENCL=OFF"
tests/ut_cuda_utils.cpp:class CudaUtilsTest : public ::testing::Test
tests/ut_cuda_utils.cpp:  typedef vc::Image2DManaged<float, vc::TargetDeviceCUDA> GPUImage;
tests/ut_cuda_utils.cpp:  CudaUtilsTest()
tests/ut_cuda_utils.cpp:    input_buf_ = std::make_unique<GPUImage>(imwidth_, imheight_);
tests/ut_cuda_utils.cpp:    downsampled_gpu_ = std::make_unique<GPUImage>(imwidth_/2, imheight_/2);
tests/ut_cuda_utils.cpp:  std::unique_ptr<GPUImage> input_buf_;
tests/ut_cuda_utils.cpp:  std::unique_ptr<GPUImage> downsampled_gpu_;
tests/ut_cuda_utils.cpp:TEST_F(CudaUtilsTest, Downsample)
tests/ut_cuda_utils.cpp:    df::GaussianBlurDown(*input_buf_, *downsampled_gpu_);
tests/ut_cuda_utils.cpp:    if (i != 0) // skip first run as cuda is warming up
tests/ut_cuda_utils.cpp:  downsampled_cpu_->copyFrom(*downsampled_gpu_);
tests/ut_cuda_utils.cpp:TEST_F(CudaUtilsTest, SobelGradients)
tests/ut_cuda_utils.cpp:  vc::Image2DManaged<Eigen::Matrix<float,1,2>, vc::TargetDeviceCUDA> sobel_gpu(imwidth_, imheight_);
tests/ut_cuda_utils.cpp:    df::SobelGradients(*input_buf_, sobel_gpu);
tests/ut_cuda_utils.cpp:    if (i != 0) // skip first run as cuda is warming up
tests/ut_cuda_utils.cpp:  sobel_cpu.copyFrom(sobel_gpu);
tests/main.cpp:#include "cuda_context.h"
tests/main.cpp:  cuda::Init();
tests/ut_se3aligner.cpp:    img0gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
tests/ut_se3aligner.cpp:    img1gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
tests/ut_se3aligner.cpp:    dpt0gpu_ = std::make_unique<ImageBuffer<vc::TargetDeviceCUDA>>(w, h);
tests/ut_se3aligner.cpp:    grad1gpu_ = std::make_unique<GradientBuffer<vc::TargetDeviceCUDA>>(w, h);
tests/ut_se3aligner.cpp:    img0gpu_->copyFrom(img0);
tests/ut_se3aligner.cpp:    img1gpu_->copyFrom(img1);
tests/ut_se3aligner.cpp:    dpt0gpu_->copyFrom(dpt0);
tests/ut_se3aligner.cpp:    df::SobelGradients(*img1gpu_, *grad1gpu_);
tests/ut_se3aligner.cpp:    return aligner_.RunStep(se3, cam_, *img0gpu_, *img1gpu_, *dpt0gpu_, *grad1gpu_);
tests/ut_se3aligner.cpp:  void RunAlignerWarp(const SE3T& se3, vc::Image2DView<Scalar, vc::TargetDeviceGPU>& img2gpu)
tests/ut_se3aligner.cpp:    aligner_.Warp(se3, cam_, *img0gpu_, *img1gpu_, *dpt0gpu_, img2gpu);
tests/ut_se3aligner.cpp:  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> img0gpu_;
tests/ut_se3aligner.cpp:  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> img1gpu_;
tests/ut_se3aligner.cpp:  std::unique_ptr<ImageBuffer<vc::TargetDeviceCUDA>> dpt0gpu_;
tests/ut_se3aligner.cpp:  std::unique_ptr<GradientBuffer<vc::TargetDeviceCUDA>> grad1gpu_;
tests/ut_se3aligner.cpp://               << " inliers:" << result.inliers / (TypeParam) this->img0gpu_->area() * 100 << "%";
tests/ut_se3aligner.cpp://     vc::Image2DManaged<TypeParam, vc::TargetDeviceCUDA> img2gpu(w, h);
tests/ut_se3aligner.cpp://     this->RunAlignerWarp(se3, img2gpu);
tests/ut_se3aligner.cpp://     img2.copyFrom(img2gpu);
tests/CMakeLists.txt:  ut_cuda_utils.cpp
tests/CMakeLists.txt:  df_cuda
tests/ut_sfmaligner.cpp:#include "cuda_context.h"
tests/ut_sfmaligner.cpp:  typedef vc::Image2DManaged<float, vc::TargetDeviceGPU> ImageBufGpu;
tests/ut_sfmaligner.cpp:    cuda::Init();
tests/ut_sfmaligner.cpp:    cuda::ScopedContextPop pop;
tests/ut_sfmaligner.cpp:    img0_gpu_ = std::make_shared<ImageBufGpu>(width, height);
tests/ut_sfmaligner.cpp:    img1_gpu_ = std::make_shared<ImageBufGpu>(width, height);
tests/ut_sfmaligner.cpp:    // copy images to gpu
tests/ut_sfmaligner.cpp:    img0_gpu_->copyFrom(*img0buf_);
tests/ut_sfmaligner.cpp:    img1_gpu_->copyFrom(*img1buf_);
tests/ut_sfmaligner.cpp:  std::shared_ptr<ImageBufGpu> img0_gpu_;
tests/ut_sfmaligner.cpp:  std::shared_ptr<ImageBufGpu> img1_gpu_;
tests/ut_sfmaligner.cpp:    cuda::ScopedContextPop pop;
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> img0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> img1_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);
tests/ut_sfmaligner.cpp:  auto gpu_result = aligner_->RunStep(pose0, pose1, code, cam_, img0_gpu, img1_gpu,
tests/ut_sfmaligner.cpp:                                      dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);
tests/ut_sfmaligner.cpp:  // gpu and cpu buffers for network output
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);
tests/ut_sfmaligner.cpp:    cuda::ScopedContextPop pop;
tests/ut_sfmaligner.cpp:  // copy network results to GPU
tests/ut_sfmaligner.cpp:  pyr_prx_gpu.copyFrom(pyr_prx);
tests/ut_sfmaligner.cpp:  pyr_jac_gpu.copyFrom(pyr_jac);
tests/ut_sfmaligner.cpp:  std0_gpu.copyFrom(pyr_std[0]);
tests/ut_sfmaligner.cpp:  // convert proximity to depth (done on GPU) and download to CPU
tests/ut_sfmaligner.cpp:  df::UpdateDepth<float,CS,vc::Buffer2DView<float, vc::TargetDeviceCUDA>>(code, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);
tests/ut_sfmaligner.cpp:  dpt0.copyFrom(dpt0_gpu);
tests/ut_sfmaligner.cpp:  df::SobelGradients(*img1_gpu_, grad1_gpu);
tests/ut_sfmaligner.cpp:  grad1.copyFrom(grad1_gpu);
tests/ut_sfmaligner.cpp:  /// calculate the GPU result
tests/ut_sfmaligner.cpp:  auto gpu_result = aligner_->RunStep(pose0, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
tests/ut_sfmaligner.cpp:                                      dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_sfmaligner.cpp:  /// compare GPU and CPU inliers and hessian
tests/ut_sfmaligner.cpp:  EXPECT_EQ(cpu_result.inliers, gpu_result.inliers);
tests/ut_sfmaligner.cpp:  EXPECT_NE(gpu_result.inliers, 0);
tests/ut_sfmaligner.cpp:  ReductionItem::HessianType::DenseMatrixType H_gpu = gpu_result.JtJ.toDenseMatrix();
tests/ut_sfmaligner.cpp:  df::CompareWithTol(H_cpu, H_gpu, 1e-1);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> dpt0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> vld0_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<Grad, vc::TargetDeviceCUDA>  grad1_gpu(width, height);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> std0_gpu(width, height);
tests/ut_sfmaligner.cpp:  // gpu and cpu buffers for network output
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_prx_gpu(pyrlevels, width, height);
tests/ut_sfmaligner.cpp:  vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA> pyr_jac_gpu(pyrlevels, width*codesize, height);
tests/ut_sfmaligner.cpp:    cuda::ScopedContextPop pop;
tests/ut_sfmaligner.cpp:  df::SobelGradients(*img1_gpu_, grad1_gpu);
tests/ut_sfmaligner.cpp:  // copy network outputs to GPU
tests/ut_sfmaligner.cpp:  pyr_prx_gpu.copyFrom(pyr_prx);
tests/ut_sfmaligner.cpp:  pyr_jac_gpu.copyFrom(pyr_jac);
tests/ut_sfmaligner.cpp:  std0_gpu.copyFrom(pyr_std[0]);
tests/ut_sfmaligner.cpp:  df::UpdateDepth<float, CS, vc::Buffer2DView<float,vc::TargetDeviceCUDA>>(code, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);
tests/ut_sfmaligner.cpp:  vc::Image2DManaged<float, vc::TargetDeviceCUDA> warped_gpu(width, height);
tests/ut_sfmaligner.cpp:  se3aligner_->Warp(pose1, cam_, *img0_gpu_, *img1_gpu_, dpt0_gpu, warped_gpu);
tests/ut_sfmaligner.cpp:  warped_cpu.copyFrom(warped_gpu);
tests/ut_sfmaligner.cpp:  auto result_0 = aligner_->RunStep(pose0, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
tests/ut_sfmaligner.cpp:                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_sfmaligner.cpp:      result = aligner_->RunStep(pose0_forward, pose1, code, cam_, *img0_gpu_, *img1_gpu_,
tests/ut_sfmaligner.cpp:                                 dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_sfmaligner.cpp:    auto result = aligner_->RunStep(pose0, pose1_forward, code, cam_, *img0_gpu_, *img1_gpu_,
tests/ut_sfmaligner.cpp:                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_sfmaligner.cpp:    df::UpdateDepth<float, CS, vc::Buffer2DView<float,vc::TargetDeviceCUDA>>(code_forward, pyr_prx_gpu[0], pyr_jac_gpu[0], avg_dpt, dpt0_gpu);
tests/ut_sfmaligner.cpp:    auto result = aligner_->RunStep(pose0, pose1, code_forward, cam_, *img0_gpu_, *img1_gpu_,
tests/ut_sfmaligner.cpp:                                    dpt0_gpu, std0_gpu, vld0_gpu, pyr_jac_gpu[0], grad1_gpu);
tests/ut_decoder.cpp:#include "cuda_context.h"
tests/ut_decoder.cpp:    cuda::ScopedContextPop pop;
tests/ut_decoder.cpp:  // image to cpu and gpu buffers
README.md: * An NVIDIA GPU with CUDA
README.md: * CUDA
README.md:You will also need to install the TensorFlow C API. If you have CUDA 10.0 and cuDNN 7.5, you can simply download the pre-built binaries by following [these instructions](https://www.tensorflow.org/install/lang_c). When using a different version of CUDA or cuDNN, pre-compiled TensorFlow C API will not work and you have to [compile it from source](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/lib_package/README.md).
README.md:boost cuda gflags glew google-glog jsoncpp opencv tensorflow-cuda 
README.md:    Allows to tune the number of blocks and threads used in the core CUDA kernels by grid search and benchmarking. These parameters can be specified with the following command line options: 
Dockerfile.ubuntu:FROM nvidia/cuda:10.0-cudnn7-devel
Dockerfile.ubuntu:# Add nvidia driver settings
Dockerfile.ubuntu:ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
Dockerfile.ubuntu:ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
Dockerfile.ubuntu:ARG TFAPI_URL=https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.15.0.tar.gz
Dockerfile.ubuntu:      -DDF_CUDA_ARCH="6.1" \
CMakeLists.txt:project(deepfactors LANGUAGES CXX CUDA)
CMakeLists.txt:find_package(CUDA         QUIET REQUIRED)
scripts/docker/run_docker.sh:    --runtime=nvidia \

```
