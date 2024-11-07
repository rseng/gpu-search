# https://github.com/CompNet/Renard

```console
renard/pipeline/speaker_attribution.py:        device: Literal["cpu", "cuda", "auto"] = "auto",
renard/pipeline/speaker_attribution.py:            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renard/pipeline/corefs/corefs.py:        device: Literal["auto", "cuda", "cpu"] = "auto",
renard/pipeline/corefs/corefs.py:            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
renard/pipeline/ner.py:        device: Literal["cpu", "cuda", "auto"] = "auto",
renard/pipeline/ner.py:            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
poetry.lock:tensorflow-gpu = ["tensorflow-gpu (>=2.2.0,!=2.6.0,!=2.6.1)"]
poetry.lock:name = "nvidia-cublas-cu12"
poetry.lock:    {file = "nvidia_cublas_cu12-12.1.3.1-py3-none-manylinux1_x86_64.whl", hash = "sha256:ee53ccca76a6fc08fb9701aa95b6ceb242cdaab118c3bb152af4e579af792728"},
poetry.lock:    {file = "nvidia_cublas_cu12-12.1.3.1-py3-none-win_amd64.whl", hash = "sha256:2b964d60e8cf11b5e1073d179d85fa340c120e99b3067558f3cf98dd69d02906"},
poetry.lock:name = "nvidia-cuda-cupti-cu12"
poetry.lock:description = "CUDA profiling tools runtime libs."
poetry.lock:    {file = "nvidia_cuda_cupti_cu12-12.1.105-py3-none-manylinux1_x86_64.whl", hash = "sha256:e54fde3983165c624cb79254ae9818a456eb6e87a7fd4d56a2352c24ee542d7e"},
poetry.lock:    {file = "nvidia_cuda_cupti_cu12-12.1.105-py3-none-win_amd64.whl", hash = "sha256:bea8236d13a0ac7190bd2919c3e8e6ce1e402104276e6f9694479e48bb0eb2a4"},
poetry.lock:name = "nvidia-cuda-nvrtc-cu12"
poetry.lock:    {file = "nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-manylinux1_x86_64.whl", hash = "sha256:339b385f50c309763ca65456ec75e17bbefcbbf2893f462cb8b90584cd27a1c2"},
poetry.lock:    {file = "nvidia_cuda_nvrtc_cu12-12.1.105-py3-none-win_amd64.whl", hash = "sha256:0a98a522d9ff138b96c010a65e145dc1b4850e9ecb75a0172371793752fd46ed"},
poetry.lock:name = "nvidia-cuda-runtime-cu12"
poetry.lock:description = "CUDA Runtime native Libraries"
poetry.lock:    {file = "nvidia_cuda_runtime_cu12-12.1.105-py3-none-manylinux1_x86_64.whl", hash = "sha256:6e258468ddf5796e25f1dc591a31029fa317d97a0a94ed93468fc86301d61e40"},
poetry.lock:    {file = "nvidia_cuda_runtime_cu12-12.1.105-py3-none-win_amd64.whl", hash = "sha256:dfb46ef84d73fababab44cf03e3b83f80700d27ca300e537f85f636fac474344"},
poetry.lock:name = "nvidia-cudnn-cu12"
poetry.lock:    {file = "nvidia_cudnn_cu12-8.9.2.26-py3-none-manylinux1_x86_64.whl", hash = "sha256:5ccb288774fdfb07a7e7025ffec286971c06d8d7b4fb162525334616d7629ff9"},
poetry.lock:nvidia-cublas-cu12 = "*"
poetry.lock:name = "nvidia-cufft-cu12"
poetry.lock:    {file = "nvidia_cufft_cu12-11.0.2.54-py3-none-manylinux1_x86_64.whl", hash = "sha256:794e3948a1aa71fd817c3775866943936774d1c14e7628c74f6f7417224cdf56"},
poetry.lock:    {file = "nvidia_cufft_cu12-11.0.2.54-py3-none-win_amd64.whl", hash = "sha256:d9ac353f78ff89951da4af698f80870b1534ed69993f10a4cf1d96f21357e253"},
poetry.lock:name = "nvidia-curand-cu12"
poetry.lock:    {file = "nvidia_curand_cu12-10.3.2.106-py3-none-manylinux1_x86_64.whl", hash = "sha256:9d264c5036dde4e64f1de8c50ae753237c12e0b1348738169cd0f8a536c0e1e0"},
poetry.lock:    {file = "nvidia_curand_cu12-10.3.2.106-py3-none-win_amd64.whl", hash = "sha256:75b6b0c574c0037839121317e17fd01f8a69fd2ef8e25853d826fec30bdba74a"},
poetry.lock:name = "nvidia-cusolver-cu12"
poetry.lock:description = "CUDA solver native runtime libraries"
poetry.lock:    {file = "nvidia_cusolver_cu12-11.4.5.107-py3-none-manylinux1_x86_64.whl", hash = "sha256:8a7ec542f0412294b15072fa7dab71d31334014a69f953004ea7a118206fe0dd"},
poetry.lock:    {file = "nvidia_cusolver_cu12-11.4.5.107-py3-none-win_amd64.whl", hash = "sha256:74e0c3a24c78612192a74fcd90dd117f1cf21dea4822e66d89e8ea80e3cd2da5"},
poetry.lock:nvidia-cublas-cu12 = "*"
poetry.lock:nvidia-cusparse-cu12 = "*"
poetry.lock:nvidia-nvjitlink-cu12 = "*"
poetry.lock:name = "nvidia-cusparse-cu12"
poetry.lock:    {file = "nvidia_cusparse_cu12-12.1.0.106-py3-none-manylinux1_x86_64.whl", hash = "sha256:f3b50f42cf363f86ab21f720998517a659a48131e8d538dc02f8768237bd884c"},
poetry.lock:    {file = "nvidia_cusparse_cu12-12.1.0.106-py3-none-win_amd64.whl", hash = "sha256:b798237e81b9719373e8fae8d4f091b70a0cf09d9d85c95a557e11df2d8e9a5a"},
poetry.lock:nvidia-nvjitlink-cu12 = "*"
poetry.lock:name = "nvidia-nccl-cu12"
poetry.lock:description = "NVIDIA Collective Communication Library (NCCL) Runtime"
poetry.lock:    {file = "nvidia_nccl_cu12-2.18.1-py3-none-manylinux1_x86_64.whl", hash = "sha256:1a6c4acefcbebfa6de320f412bf7866de856e786e0462326ba1bac40de0b5e71"},
poetry.lock:name = "nvidia-nvjitlink-cu12"
poetry.lock:description = "Nvidia JIT LTO Library"
poetry.lock:    {file = "nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux1_x86_64.whl", hash = "sha256:64335a8088e2b9d196ae8665430bc6a2b7e6ef2eb877a9c735c804bd4ff6467c"},
poetry.lock:    {file = "nvidia_nvjitlink_cu12-12.3.101-py3-none-manylinux2014_aarch64.whl", hash = "sha256:211a63e7b30a9d62f1a853e19928fbb1a750e3f17a13a3d1f98ff0ced19478dd"},
poetry.lock:    {file = "nvidia_nvjitlink_cu12-12.3.101-py3-none-win_amd64.whl", hash = "sha256:1b2e317e437433753530792f13eece58f0aec21a2b05903be7bffe58a606cbd1"},
poetry.lock:name = "nvidia-nvtx-cu12"
poetry.lock:description = "NVIDIA Tools Extension"
poetry.lock:    {file = "nvidia_nvtx_cu12-12.1.105-py3-none-manylinux1_x86_64.whl", hash = "sha256:dc21cf308ca5691e7c04d962e213f8a4aa9bbfa23d95412f452254c2caeb09e5"},
poetry.lock:    {file = "nvidia_nvtx_cu12-12.1.105-py3-none-win_amd64.whl", hash = "sha256:65f4d98982b31b60026e0e6de73fbdfc09d08a96f4656dd3665ca616a11e1e82"},
poetry.lock:cuda = ["cupy (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda-autodetect = ["cupy-wheel (>=11.0.0,<13.0.0)"]
poetry.lock:cuda100 = ["cupy-cuda100 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda101 = ["cupy-cuda101 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda102 = ["cupy-cuda102 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda110 = ["cupy-cuda110 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda111 = ["cupy-cuda111 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda112 = ["cupy-cuda112 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda113 = ["cupy-cuda113 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda114 = ["cupy-cuda114 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda115 = ["cupy-cuda115 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda116 = ["cupy-cuda116 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda117 = ["cupy-cuda117 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda11x = ["cupy-cuda11x (>=11.0.0,<13.0.0)"]
poetry.lock:cuda80 = ["cupy-cuda80 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda90 = ["cupy-cuda90 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda91 = ["cupy-cuda91 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda92 = ["cupy-cuda92 (>=5.0.0b4,<13.0.0)"]
poetry.lock:cuda = ["cupy (>=5.0.0b4)"]
poetry.lock:cuda100 = ["cupy-cuda100 (>=5.0.0b4)"]
poetry.lock:cuda101 = ["cupy-cuda101 (>=5.0.0b4)"]
poetry.lock:cuda102 = ["cupy-cuda102 (>=5.0.0b4)"]
poetry.lock:cuda110 = ["cupy-cuda110 (>=5.0.0b4)"]
poetry.lock:cuda111 = ["cupy-cuda111 (>=5.0.0b4)"]
poetry.lock:cuda112 = ["cupy-cuda112 (>=5.0.0b4)"]
poetry.lock:cuda80 = ["cupy-cuda80 (>=5.0.0b4)"]
poetry.lock:cuda90 = ["cupy-cuda90 (>=5.0.0b4)"]
poetry.lock:cuda91 = ["cupy-cuda91 (>=5.0.0b4)"]
poetry.lock:cuda92 = ["cupy-cuda92 (>=5.0.0b4)"]
poetry.lock:cuda = ["cupy (>=5.0.0b4)"]
poetry.lock:cuda-autodetect = ["cupy-wheel (>=11.0.0)"]
poetry.lock:cuda100 = ["cupy-cuda100 (>=5.0.0b4)"]
poetry.lock:cuda101 = ["cupy-cuda101 (>=5.0.0b4)"]
poetry.lock:cuda102 = ["cupy-cuda102 (>=5.0.0b4)"]
poetry.lock:cuda110 = ["cupy-cuda110 (>=5.0.0b4)"]
poetry.lock:cuda111 = ["cupy-cuda111 (>=5.0.0b4)"]
poetry.lock:cuda112 = ["cupy-cuda112 (>=5.0.0b4)"]
poetry.lock:cuda113 = ["cupy-cuda113 (>=5.0.0b4)"]
poetry.lock:cuda114 = ["cupy-cuda114 (>=5.0.0b4)"]
poetry.lock:cuda115 = ["cupy-cuda115 (>=5.0.0b4)"]
poetry.lock:cuda116 = ["cupy-cuda116 (>=5.0.0b4)"]
poetry.lock:cuda117 = ["cupy-cuda117 (>=5.0.0b4)"]
poetry.lock:cuda11x = ["cupy-cuda11x (>=11.0.0)"]
poetry.lock:cuda80 = ["cupy-cuda80 (>=5.0.0b4)"]
poetry.lock:cuda90 = ["cupy-cuda90 (>=5.0.0b4)"]
poetry.lock:cuda91 = ["cupy-cuda91 (>=5.0.0b4)"]
poetry.lock:cuda92 = ["cupy-cuda92 (>=5.0.0b4)"]
poetry.lock:description = "Tensors and Dynamic neural networks in Python with strong GPU acceleration"
poetry.lock:nvidia-cublas-cu12 = {version = "12.1.3.1", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cuda-cupti-cu12 = {version = "12.1.105", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cuda-nvrtc-cu12 = {version = "12.1.105", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cuda-runtime-cu12 = {version = "12.1.105", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cudnn-cu12 = {version = "8.9.2.26", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cufft-cu12 = {version = "11.0.2.54", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-curand-cu12 = {version = "10.3.2.106", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cusolver-cu12 = {version = "11.4.5.107", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-cusparse-cu12 = {version = "12.1.0.106", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-nccl-cu12 = {version = "2.18.1", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}
poetry.lock:nvidia-nvtx-cu12 = {version = "12.1.105", markers = "platform_system == \"Linux\" and platform_machine == \"x86_64\""}

```
