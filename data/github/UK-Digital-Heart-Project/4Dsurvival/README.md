# https://github.com/UK-Digital-Heart-Project/4Dsurvival

```console
Dockerfile:# lisurui6/4dsurvival-gpu:1.1
Dockerfile:FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
README.md:### Download 4D*survival* Docker image (GPU)
README.md:    docker pull lisurui6/4dsurvival-gpu:1.1
README.md:should show `lisurui6/4dsurvival-gpu:1.0` on the list of Docker images on your local system
README.md:To be able to utilise GPU in docker container, run
README.md:    nvidia-docker run -it lisurui6/4dsurvival:1.1
README.md:Dockerfile for build the GPU image is described in [Dockerfile](Dockerfile).
README.md:    nvidia-docker run -it -v /path-to-your-data:/data -v /path-to-your-4ds-code:/4DSurvival -v /path-to-your-experiment-dir:/exp-dir/ lisurui6/4dsurvival:1.1
README.md:    CUDA_VISIBLE_DEVICES=0 python demo_validate_nn.py -c /exp-dir/exp-name.conf
survival4D/nn/torch/models.py:        x = torch.from_numpy(x).cuda().float()
survival4D/nn/torch/models.py:    torch.cuda.empty_cache()
survival4D/nn/torch/__init__.py:    dataset = TensorDataset(torch.from_numpy(X_tr).cuda(), torch.from_numpy(E_tr).cuda(), torch.from_numpy(TM_tr).cuda())
survival4D/nn/torch/__init__.py:    model.cuda()

```
