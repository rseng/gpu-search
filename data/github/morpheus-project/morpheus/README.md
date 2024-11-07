# https://github.com/morpheus-project/morpheus

```console
README.rst:to support the GPU accelerated version of TensorFlow, which has a different
README.rst:Morpheus has two main flavors of Docker Image: ``gpu`` for the GPU enabled version
README.rst:For GPU support:
README.rst:    docker run --runtime=nvidia -it morpheusastro/morpheus:lastest-gpu
README.rst:jobs or GPU jobs. Importantly, you cannot specify both at the same time.
README.rst:**GPUS**
README.rst:The ``gpus`` argument should be a list of integers that are the ids assigned to
README.rst:the GPUS to be used. These ids can be found by using ``nvidia-smi``.
README.rst:    classified = Classifier.classify(h=h, j=j, v=v, z=z, gpus=[0,1])
README.rst:jobs or GPU jobs. Importantly, you cannot specify both at the same time.
README.rst:**GPUS**
README.rst:The ``gpus`` optional flag should be a comma-separated list of ids for the
README.rst:GPUS to be used. These ids can be found by using ``nvidia-smi``.
README.rst:    morpheus h.fits j.fits v.fits z.fits --gpus 0,1
travis_deploy.py:6. Build docker image for cpu and gpu
travis_deploy.py:    docker_gpu = os.path.join(LOCAL, "docker/Dockerfile.gpu")
travis_deploy.py:    print("Building docker GPU")
travis_deploy.py:        f"docker build --no-cache -t morpheusastro/morpheus:{docker_ver}-gpu -t morpheusastro/morpheus:latest-gpu -f {docker_gpu} ."
travis_deploy.py:    os.system(f"docker push morpheusastro/morpheus:{docker_ver}-gpu")
travis_deploy.py:    os.system(f"docker push morpheusastro/morpheus:latest-gpu")
morpheus/tests/test_classifier.py:    def test_validate_parallel_params_raises_cpus_gpus():
morpheus/tests/test_classifier.py:        Throws ValueError for passing values for both cpus an gpus.
morpheus/tests/test_classifier.py:        gpus = [0]
morpheus/tests/test_classifier.py:            Classifier._validate_parallel_params(gpus=gpus, cpus=cpus)
morpheus/tests/test_classifier.py:    def test_validate_parallel_params_raises_single_gpu():
morpheus/tests/test_classifier.py:        Throws ValueError for passing a single gpu.
morpheus/tests/test_classifier.py:        gpus = [0]
morpheus/tests/test_classifier.py:            Classifier._validate_parallel_params(gpus=gpus)
morpheus/tests/test_classifier.py:        Throws ValueError for passing a single gpu.
morpheus/tests/test_cli.py:    def test_gpus():
morpheus/tests/test_cli.py:        """Tests _gpus."""
morpheus/tests/test_cli.py:        gpus = "1,2,3"
morpheus/tests/test_cli.py:        assert [1, 2, 3] == cli._gpus(gpus)
morpheus/tests/test_cli.py:    def test_gpus_raises():
morpheus/tests/test_cli.py:        """Test _gpus raises ValueError for passing single gpus."""
morpheus/tests/test_cli.py:        gpus = "1"
morpheus/tests/test_cli.py:            cli._gpus(gpus)
morpheus/tests/test_cli.py:    def test_parse_args_raises_cpus_gpus():
morpheus/tests/test_cli.py:        """test _parse_args raise ValueError for passing cpus and gpus."""
morpheus/tests/test_cli.py:        cli_args += "--cpus 3 --gpus 1,2,3"
morpheus/tests/test_cli.py:        cli_args += "--cpus 3 --gpus 1,2,3"
morpheus/__main__.py:def _gpus(value):
morpheus/__main__.py:    gpus = [int(v) for v in value.split(",")]
morpheus/__main__.py:    gpu_err = "--gpus option requires more than one GPU ID. If you are trying "
morpheus/__main__.py:    gpu_err += "to select a single gpu to use the CUDA_VISIBLE_DEVICES "
morpheus/__main__.py:    gpu_err += "environment variable. For more information: "
morpheus/__main__.py:    gpu_err += "https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/"
morpheus/__main__.py:    if len(gpus) < 2:
morpheus/__main__.py:        raise ValueError(gpu_err)
morpheus/__main__.py:    return gpus
morpheus/__main__.py:    # parallel gpu
morpheus/__main__.py:    gpus_desc = "Optional flag for classifying an image in parallel with multiple "
morpheus/__main__.py:    gpus_desc += "GPUs. Should be comma seperated ints like: 1,3 or 0,1,2 no spaces."
morpheus/__main__.py:    gpus_desc += "DO NOT use this flag for a single GPU classification. "
morpheus/__main__.py:    gpus_desc += "Use the CUDA_VISIBLE_DEVICES enironment variable to select a "
morpheus/__main__.py:    gpus_desc += "GPU for morpheus to use."
morpheus/__main__.py:    parser.add_argument("--gpus", type=_gpus, help=gpus_desc)
morpheus/__main__.py:    print(args.cpus, args.gpus)
morpheus/__main__.py:    if args.cpus and args.gpus:
morpheus/__main__.py:        raise ValueError("Both --cpus and --gpus were indicated. Choose only one.")
morpheus/__main__.py:            gpus=args.gpus,
morpheus/__main__.py:            gpus=args.gpus,
morpheus/__main__.py:            gpus=args.gpus,
morpheus/__main__.py:            gpus=args.gpus,
morpheus/__main__.py:            gpus=args.gpus,
morpheus/classifier.py:        gpus: List[int] = None,
morpheus/classifier.py:            gpus (List[int]): The GPU ids to use for parallel classification
morpheus/classifier.py:                              the ids can be found using ``nvidia-smi``
morpheus/classifier.py:            ValueError if both gpus and cpus are given
morpheus/classifier.py:        workers, is_gpu = Classifier._validate_parallel_params(gpus, cpus)
morpheus/classifier.py:                workers, is_gpu, out_dir, parallel_check_interval
morpheus/classifier.py:            workers (List[int]): A list of worker ID's that can either be CUDA GPU
morpheus/classifier.py:            workers (List[int]): A list of worker ID's that can either be CUDA GPU
morpheus/classifier.py:        workers: List[int], is_gpu: bool, out_dir: str, parallel_check_interval: float
morpheus/classifier.py:            is_gpu (bool): if True the worker ID's belong to NVIDIA GPUs and will
morpheus/classifier.py:                           be used as an argument in CUDA_VISIBLE_DEVICES. If False,
morpheus/classifier.py:            parallel_check_interval (float): If gpus are given, then this is the number
morpheus/classifier.py:            if is_gpu:
morpheus/classifier.py:                cmd_string = f"CUDA_VISIBLE_DEVICES={worker} python main.py"
morpheus/classifier.py:                cmd_string = f"CUDA_VISIBLE_DEVICES=-1 python main.py"
morpheus/classifier.py:        gpus: List[int] = None, cpus: int = None
morpheus/classifier.py:            gpus (List[int]): A list of the CUDA gpu ID's to use for a
morpheus/classifier.py:            wheter or not the ids belong to GPUS
morpheus/classifier.py:            ValueError if both cpus and gpus are not None
morpheus/classifier.py:        if (gpus is not None) and (cpus is not None):
morpheus/classifier.py:            raise ValueError("Please only give a value cpus or gpus, not both.")
morpheus/classifier.py:        if (gpus is None) and (cpus is None):
morpheus/classifier.py:        if gpus is not None:
morpheus/classifier.py:            if len(gpus) == 1:
morpheus/classifier.py:                err = "Only one gpus indicated. If you are trying to select "
morpheus/classifier.py:                err += "a single gpu, then use the CUDA_VISIBLE_DEVICES environment "
morpheus/classifier.py:                err += "https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/"
morpheus/classifier.py:                return gpus, True
docker/Dockerfile.gpu:# Adapted from https://github.com/samuelcolvin/tensorflow-gpu-py36/blob/master/Dockerfile
docker/Dockerfile.gpu:# This should be much easier after tensorflow support python 3.6 with CUDA 10
docker/Dockerfile.gpu:# docker build --no-cache -t morpheusastro/morpheus:0.3-gpu -f Dockerfile.gpu .
docker/Dockerfile.gpu:# docker push morpheusastro/morpheus:0.3-gpu
docker/Dockerfile.gpu:FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
docker/Dockerfile.gpu:RUN python3.6 -m pip install --no-cache-dir -U tensorflow-gpu

```
