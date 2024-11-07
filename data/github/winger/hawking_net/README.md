# https://github.com/winger/hawking_net

```console
main.cu:int gpu0;
main.cu:#define CUDA_CHECK(expr) do {                                                                      \
main.cu:  cudaError_t rc = (expr);                                                                         \
main.cu:  if (rc != cudaSuccess) {                                                                      \
main.cu:        printf("CUDA error %d '%s' - %s:%d: %s", rc, cudaGetErrorString(rc),  __FILE__, __LINE__, #expr); \
main.cu:    // CUDA_CHECK(cudaMemPrefetchAsync(data, n_data * sizeof(point), gpu0));
main.cu:    // CUDA_CHECK(cudaMemPrefetchAsync(queries, n_queries * sizeof(query), gpu0));
main.cu:    CUDA_CHECK(cudaDeviceSynchronize());
main.cu:    CUDA_CHECK(cudaGetDevice(&gpu0));
main.cu:    CUDA_CHECK(cudaMallocManaged(&data, base.Npix() * sizeof(point)));
main.cu:    CUDA_CHECK(cudaMallocManaged(&queries, queries_base.Npix() * sizeof(query)));
main.cu:    CUDA_CHECK(cudaFree(data));
main.cu:    CUDA_CHECK(cudaFree(queries));
requirements.txt:cupy-cuda12x==12.2.0
requirements.txt:nvidia-cublas-cu11==11.10.3.66
requirements.txt:nvidia-cuda-cupti-cu11==11.7.101
requirements.txt:nvidia-cuda-nvrtc-cu11==11.7.99
requirements.txt:nvidia-cuda-runtime-cu11==11.7.99
requirements.txt:nvidia-cudnn-cu11==8.5.0.96
requirements.txt:nvidia-cufft-cu11==10.9.0.58
requirements.txt:nvidia-curand-cu11==10.2.10.91
requirements.txt:nvidia-cusolver-cu11==11.4.0.1
requirements.txt:nvidia-cusparse-cu11==11.7.4.91
requirements.txt:nvidia-nccl-cu11==2.14.3
requirements.txt:nvidia-nvtx-cu11==11.7.91
hawking_points.py:# model = Net().cuda()
hawking_points.py:model = PreActResNet18(in_planes=1, num_classes=1).cuda()
hawking_points.py:    return hp.cuda()
hawking_points.py:    #         batch = batch.cuda()
hawking_points.py:            batch = batch.cuda()
hawking_points.py:            batch = batch.cuda() * 1e6
hawking_points.py:        batch = batch.cuda() * 1e6
hawking_points.py:        # print(model(batch.cuda()[:, None]).shape)

```
