# https://github.com/supernnova/SuperNNova

```console
run_onthefly.py:        "--device", type=str, default="cpu", help="device to be used [cuda,cpu]"
python/supernnova/visualization/prediction_distribution.py:    if settings.use_cuda:
python/supernnova/visualization/prediction_distribution.py:        X_tensor.cuda()
python/supernnova/visualization/early_prediction.py:    if settings.use_cuda:
python/supernnova/visualization/early_prediction.py:        X_tensor.cuda()
python/supernnova/paper/superNNova_plots.py:    df_gpu = df[df.device == "gpu"]
python/supernnova/paper/superNNova_plots.py:    gpu_is_available = len(df_gpu) > 0
python/supernnova/paper/superNNova_plots.py:    if gpu_is_available:
python/supernnova/paper/superNNova_plots.py:        # Space bars by 2 units to leave room for gpu
python/supernnova/paper/superNNova_plots.py:        idxs_gpu = idxs_cpu + 1
python/supernnova/paper/superNNova_plots.py:    if gpu_is_available:
python/supernnova/paper/superNNova_plots.py:        for i in range(len(idxs_gpu)):
python/supernnova/paper/superNNova_plots.py:            label = "GPU" if i == 0 else None
python/supernnova/paper/superNNova_plots.py:                idxs_gpu[i],
python/supernnova/paper/superNNova_plots.py:                df_gpu["Supernova_per_s"].values[i],
python/supernnova/training/bayesian_rnn.py:        self.use_cuda = settings.use_cuda
python/supernnova/training/bayesian_rnn.py:            # on the GPU
python/supernnova/training/bayesian_rnn.py:            # on the GPU
python/supernnova/training/bayesian_rnn.py:            # on the GPU
python/supernnova/training/bayesian_rnn.py:    if x.is_cuda:
python/supernnova/training/bayesian_rnn.py:            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma1]).cuda()
python/supernnova/training/bayesian_rnn.py:            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma2]).cuda()
python/supernnova/training/vanilla_rnn.py:        self.use_cuda = settings.use_cuda
python/supernnova/training/bayesian_rnn_2.py:        self.use_cuda = settings.use_cuda
python/supernnova/training/bayesian_rnn_2.py:    if x.is_cuda:
python/supernnova/training/bayesian_rnn_2.py:            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma1]).cuda()
python/supernnova/training/bayesian_rnn_2.py:            torch.tensor([0.0]).cuda(), torch.tensor([prior.sigma2]).cuda()
python/supernnova/training/train_rnn.py:    # Prepare for GPU if required
python/supernnova/training/train_rnn.py:    if settings.use_cuda:
python/supernnova/training/train_rnn.py:        rnn.cuda()
python/supernnova/training/train_rnn.py:        criterion.cuda()
python/supernnova/training/train_rnn.py:    # Prepare for GPU if required
python/supernnova/training/train_rnn.py:    if settings.use_cuda:
python/supernnova/training/train_rnn.py:        rnn.cuda()
python/supernnova/training/train_rnn.py:        criterion.cuda()
python/supernnova/training/train_rnn.py:    # Prepare for GPU if required
python/supernnova/training/train_rnn.py:    if settings.use_cuda:
python/supernnova/training/train_rnn.py:        rnn.cuda()
python/supernnova/training/train_rnn.py:        criterion.cuda()
python/supernnova/training/train_rnn.py:    # Prepare for GPU if required
python/supernnova/training/train_rnn.py:    if settings.use_cuda:
python/supernnova/training/train_rnn.py:        rnn.cuda()
python/supernnova/training/train_rnn.py:        criterion.cuda()
python/supernnova/training/variational_rnn.py:        self.use_cuda = settings.use_cuda
python/supernnova/conf.py:    "--use_cuda",
python/supernnova/conf.py:        "--use_cuda", action="store_true", help="Use GPU (pytorch backend only)"  # test
python/supernnova/conf.py:    cli_args["use_cuda"] = settings.use_cuda
python/supernnova/validation/validate_onthefly.py:        device (str): wehter to use cuda or cpu
python/supernnova/validation/validate_onthefly.py:    settings.use_cuda = True if "cuda" in str(device) else False
python/supernnova/utils/training_utils.py:    # Move data to GPU if required
python/supernnova/utils/training_utils.py:    if settings.use_cuda:
python/supernnova/utils/training_utils.py:        X_tensor = X_tensor.cuda()
python/supernnova/utils/training_utils.py:        target_tensor = torch.LongTensor(list_target).cuda()
python/supernnova/utils/experiment_settings.py:        if self.use_cuda:
python/supernnova/utils/experiment_settings.py:            self.device = "cuda"
cli/run.py:        if torch.cuda.is_available():
cli/run.py:            torch.cuda.manual_seed_all(settings.seed)
.pre-commit-config.yaml:  - repo: https://github.com/floatingpurr/sync_with_poetry
docs/visualization/index.rst:    cd env && python launch_docker.py (--use_cuda optional)
docs/installation/five_minute_guide_module.rst:	c) Install packages manually. Inspect ``env/conda_env.yml`` (or ``env/conda_gpu_env.yml`` when using cuda) and ``pyproject.toml`` for the list of packages we use.
docs/installation/system.rst:	GPU        : GeForce GTX 1080
docs/installation/FAQ.rst:If you have a GPU, you can activate training on GPU with the ``--use_cuda`` flag.
docs/installation/env.rst:    conda env create -f env/conda_gpu_env.yml
docs/installation/env.rst:if you want to install ``PyTorch`` with cuda support.
docs/installation/env.rst:    conda activate supernnova-cuda
docs/installation/env.rst:if you create environment from "conda_gpu_env.yml".
docs/installation/env.rst:where ``image`` is one of ``cpu`` or ``gpu`` (for the latest supported CUDA version; currently 12.3.1) or ``gpu9`` (for cuda 9.0)
docs/installation/env.rst:- Add ``--image image`` where image is ``cpu`` or ``gpu`` (latest version) or ``gpu9`` (for cuda 9)
docs/installation/five_minute_guide.rst:	c) Install packages manually. Inspect ``env/conda_env.yml`` (or ``env/conda_gpu_env.yml`` when using cuda) and ``pyproject.toml`` for the list of packages we use.
docs/data/index.rst:    cd env && python launch_docker.py (--use_cuda optional)
docs/configuration/index.rst:--use_cuda                bool          Use GPU
docs/paper/index.rst:With a GPU and a ``--batch_size = 128`` (default) this takes around two weeks. If you increase ``batch_size`` it may be reduced to a couple of days but performance can be slightly reduced.
docs/training/index.rst:    cd env && python launch_docker.py (--use_cuda optional)
docs/validation/index.rst:    cd env && python launch_docker.py (--use_cuda optional)
docs/notes_for_developers.md:    $ conda env create -f env/conda_gpu_env.yml
docs/notes_for_developers.md:    $ conda activate supernnova-cuda
docs/notes_for_developers.md:    if you use `PyTorch` with CUDA support.
docs/notes_for_developers.md:    As shown above, we establish a Python environment using either `env/conda_env.yml` or `conda_gpu_env.yml`. These configurations manage the Python version and directly handle our core dependency, `PyTorch`, as well as the installation of the `poetry` package, which is used to manage all other Python dependencies listed in `pyproject.toml`. This setup makes it easiler to manage different versions of `PyTorch` and CUDA support, streamlining the upgrade process for future versions as part of our CI/CD best practices. 
docs/notes_for_developers.md:The git hooks are defined in the `.pre-commit-config.yaml` file.  Specific revisions for many of the tools listed should be managed with Poetry, with syncing managed with the [sync_with_poetry](https://github.com/floatingpurr/sync_with_poetry) hook.  Developers should take care not to use git hooks to *enforce* any project policies.  That should all be done within the continuous integration workflows.  Instead: these should just be quality-of-life checks that fix minor issues or prevent the propagation of quick-and-easy-to-detect problems which would otherwise be caught by the CI later with considerably more latency.  Furthermore, ensure that the checks performed here are consistant between the hooks and the CI.  For example: make sure that any linting/code quality checks are executed with the same tools and options.
configs_yml/classify.yml:use_cuda: False
tests/onthefly_model/cli_args.json:        "use_cuda": false,
tests/onthefly_model/cli_args.json:    "use_cuda": false,
conda_gpu_env.yml:name: supernnova-cuda
conda_gpu_env.yml:  - nvidia
conda_gpu_env.yml:  - pytorch-cuda=11.8
env/launch_docker.py:        choices=["cpu", "gpu", "gpu10"],
env/launch_docker.py:        help="Use which image gpu or cpu",
env/launch_docker.py:    cmd += " --gpus all " if "gpu" in args.image else ""
env/launch_docker.py:        print("You may not have a GPU.")
env/Dockerfile:# Base image for 'gpu9' builds, using CUDA v9.0
env/Dockerfile:FROM nvcr.io/nvidia/cuda:9.0-devel-ubuntu16.04 as gpu9
env/Dockerfile:# Use the GPU version of the Conda environment file
env/Dockerfile:ENV CONDA_ENV_FILE="conda_gpu_env.yml"
env/Dockerfile:ENV CONDA_ENV_NAME="supernnova-cuda"
env/Dockerfile:# Base image for 'gpu' builds, using a recent CUDA version
env/Dockerfile:# Note that Nvidia no longer supports a 'latest' tag,
env/Dockerfile:FROM nvcr.io/nvidia/cuda:12.3.1-devel-ubuntu22.04 as gpu
env/Dockerfile:# Use the GPU version of the Conda environment file
env/Dockerfile:ENV CONDA_ENV_FILE="conda_gpu_env.yml"
env/Dockerfile:ENV CONDA_ENV_NAME="supernnova-cuda"
env/Dockerfile:# & CUDA versions while using a modern Python version.  In the
env/verify_cuda_support.py:print("CUDA available: ", torch.cuda.is_available())
env/verify_cuda_support.py:print("CUDA version: ", torch.version.cuda)
env/conda_gpu_env.yml:name: supernnova-cuda
env/conda_gpu_env.yml:  - nvidia
env/conda_gpu_env.yml:  - pytorch-cuda=11.8
env/archive/conda_env_gpu_linux64.txt:https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-9.0-h13b8566_0.tar.bz2
env/archive/conda_env_gpu_linux64.txt:https://repo.anaconda.com/pkgs/main/linux-64/libgpuarray-0.7.6-h14c3975_0.tar.bz2
env/archive/conda_env_gpu_linux64.txt:https://repo.anaconda.com/pkgs/main/linux-64/pygpu-0.7.6-py36h035aef0_0.tar.bz2
env/archive/conda_env_cpu_osx-64.txt:# https://conda.anaconda.org/pytorch/osx-64/pytorch-0.4.1-py36_cuda0.0_cudnn0.0_1.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://aws-ml-conda-pre-prod-ec2.s3.us-west-2.amazonaws.com/linux-64/aws-ofi-nccl-dlc-1.5.0-aws_0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-cudart-11.8.89-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-cupti-11.8.87-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-nvrtc-11.8.89-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-nvtx-11.8.86-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcublas-11.11.3.6-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcufft-10.9.0.58-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcufile-1.4.0.31-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcurand-10.3.0.86-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcusolver-11.4.1.48-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libcusparse-11.7.5.86-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libnpp-11.8.0.86-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/libnvjpeg-11.9.0.86-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://aws-ml-conda-pre-prod-ec2.s3.us-west-2.amazonaws.com/noarch/pytorch-mutex-1.0-cuda.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-libraries-11.8.0-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://conda.anaconda.org/nvidia/label/cuda-11.8.0/linux-64/cuda-runtime-11.8.0-0.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://aws-ml-conda-pre-prod-ec2.s3.us-west-2.amazonaws.com/linux-64/pytorch-cuda-11.8-h7e8668a_3.tar.bz2
env/archive/conda_env_gpu_linux64_2023.txt:https://aws-ml-conda-pre-prod-ec2.s3.us-west-2.amazonaws.com/linux-64/pytorch-2.0.0-aws_py3.10_cuda11.8_cudnn8.7.0_0.tar.bz2
env/archive/conda_env_cpu_macos.yml:  - pytorch=0.4.1=py37_cuda0.0_cudnn0.0_1
sandbox/rnn_mnist.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
sandbox/rnn_mnist.py:    if device == "cuda":
sandbox/rnn_mnist.py:        net.cuda()
sandbox/rnn_mnist.py:    # if torch.cuda.is_available():
sandbox/rnn_mnist.py:    #     torch.cuda.manual_seed_all(args.seed)
sandbox/mnist.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
sandbox/mnist.py:    if device == "cuda":
sandbox/mnist.py:        net.cuda()
sandbox/mnist.py:    if torch.cuda.is_available():
sandbox/mnist.py:        torch.cuda.manual_seed_all(args.seed)
sandbox/lm_rnn_gal.py:    device = torch.device("cuda" if args.cuda else "cpu")
sandbox/lm_rnn_gal.py:    parser.add_argument("--cuda", action="store_true", help="use CUDA")
sandbox/lm_rnn_gal.py:    if torch.cuda.is_available():
sandbox/lm_rnn_gal.py:        torch.cuda.manual_seed_all(args.seed)
sandbox/lm_rnn_gal.py:    if torch.cuda.is_available():
sandbox/lm_rnn_gal.py:        if not args.cuda:
sandbox/lm_rnn_gal.py:                "WARNING: You have a CUDA device, so you should probably run with --cuda"
sandbox/lm_rnn_fortunato.py:    device = torch.device("cuda" if args.cuda else "cpu")
sandbox/lm_rnn_fortunato.py:    parser.add_argument("--cuda", action="store_true", help="use CUDA")
sandbox/lm_rnn_fortunato.py:    if torch.cuda.is_available():
sandbox/lm_rnn_fortunato.py:        torch.cuda.manual_seed_all(args.seed)
sandbox/lm_rnn_fortunato.py:    if torch.cuda.is_available():
sandbox/lm_rnn_fortunato.py:        if not args.cuda:
sandbox/lm_rnn_fortunato.py:                "WARNING: You have a CUDA device, so you should probably run with --cuda"

```
