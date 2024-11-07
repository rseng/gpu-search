# https://github.com/DeepRegNet/DeepReg

```console
setup.py:        "Environment :: GPU",
Dockerfile:FROM tensorflow/tensorflow:2.3.1-gpu
docs/source/getting_started/install.rst:        Install DeepReg without GPU support.
docs/source/getting_started/install.rst:        Install DeepReg with GPU support.
docs/source/getting_started/install.rst:        Install DeepReg without GPU support.
docs/source/getting_started/install.rst:        Install DeepReg with GPU support.
docs/source/getting_started/install.rst:        Install DeepReg without GPU support.
docs/source/getting_started/install.rst:        Install DeepReg with GPU support.
docs/source/getting_started/quick_start.md:deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
docs/source/getting_started/quick_start.md:- `--gpu ""` indicates using CPU. Change to `--gpu "0"` to use the GPU at index 0.
docs/source/getting_started/quick_start.md:deepreg_predict --gpu "" --ckpt_path logs/test/save/ckpt-2 --split test
docs/source/docs/logging.md:  deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
docs/source/docs/logging.md:  DEEPREG_LOG_LEVEL=1 deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
docs/source/docs/logging.md:  deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
docs/source/docs/logging.md:  TF_CPP_MIN_LOG_LEVEL=1 deepreg_train --gpu "" --config_path config/unpaired_labeled_ddf.yaml --exp_name test
docs/source/docs/cli.md:- **GPU**:
docs/source/docs/cli.md:  `--gpu` or `-g`, specifies the index or indices of GPUs for training.
docs/source/docs/cli.md:  - `--gpu ""` for CPU only
docs/source/docs/cli.md:  - `--gpu "0"` for using only GPU 0
docs/source/docs/cli.md:  - `--gpu "0,1"` for using GPU 0 and 1.
docs/source/docs/cli.md:- **GPU memory allocation**:
docs/source/docs/cli.md:  `--gpu_allow_growth` or `-gr`, if given, TensorFlow will only grow the memory usage as
docs/source/docs/cli.md:  By default, it allocates all available GPU memory.
docs/source/docs/cli.md:  - `--gpu_allow_growth`, no extra argument is needed.
docs/source/docs/cli.md:- **GPU**:
docs/source/docs/cli.md:  `--gpu` or `-g`, specifies the index or indices of GPUs for training.
docs/source/docs/cli.md:  - `--gpu ""` for CPU only
docs/source/docs/cli.md:  - `--gpu "0"` for using only GPU 0
docs/source/docs/cli.md:  - `--gpu "0,1"` for using GPU 0 and 1.
docs/source/docs/cli.md:- **GPU memory allocation**:
docs/source/docs/cli.md:  `--gpu_allow_growth` or `-gr`, if given, TensorFlow will only grow the memory usage as
docs/source/docs/cli.md:  By default, it allocates all availables in the GPU memory.
docs/source/docs/cli.md:  - `--gpu_allow_growth`, no extra argument is needed.
docs/source/docs/cli.md:  using multiple GPUs, i.e. `n` GPUs, each GPU will have mini batch size
docs/source/docs/configuration.md:  multiple GPUs, i.e. `n` GPUs, each GPU will have mini batch size `batch_size / n`.
docs/source/tutorial/run_cluster.md:source /share/apps/source_files/cuda/cuda-10.1.source    # set up cuda for GPU
docs/source/tutorial/run_cluster.md:#$ -l gpu=true   # use gpu
docs/source/tutorial/run_cluster.md:export PATH=/share/apps/cuda-10.1/bin:/share/apps/gcc-8.3/bin:$PATH   # path for cuda, gcc
docs/source/tutorial/run_cluster.md:export LD_LIBRARY_PATH=/share/apps/cuda-10.1/lib64:/share/apps/gcc-8.3/lib64:$LD_LIBRARY_PATH   # path for cuda, gcc
docs/source/tutorial/run_cluster.md:--gpu \
docs/source/demo/readme_template.md:Here the training is launched using the GPU of index 0 with a limited number of steps
test/unit/test_train.py:            "--gpu",
test/unit/test_train.py:            "--gpu",
test/unit/test_callback.py:    if len(tf.config.list_physical_devices("gpu")) > 1:
CHANGELOG.md:- Fixed using GPU remotely
CHANGELOG.md:- Changed distribute strategy to default for <= 1 GPU.
environment.yml:  - cudatoolkit=10.1
environment.yml:      - tensorflow-gpu==2.3.1
deepreg/predict.py:    gpu: str,
deepreg/predict.py:    gpu_allow_growth: bool = True,
deepreg/predict.py:    :param gpu: which env gpu to use.
deepreg/predict.py:    :param gpu_allow_growth: whether to allocate whole GPU memory for training.
deepreg/predict.py:    if gpu is not None:
deepreg/predict.py:        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
deepreg/predict.py:        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
deepreg/predict.py:            "false" if gpu_allow_growth else "true"
deepreg/predict.py:    # use strategy to support multiple GPUs
deepreg/predict.py:    # the network is mirrored in each GPU so that we can use larger batch size
deepreg/predict.py:    num_devices = max(len(tf.config.list_physical_devices("GPU")), 1)
deepreg/predict.py:        "--gpu",
deepreg/predict.py:        help="GPU index for training."
deepreg/predict.py:        "-g for using GPU remotely"
deepreg/predict.py:        '-g "0" for using GPU 0'
deepreg/predict.py:        '-g "0,1" for using GPU 0 and 1.',
deepreg/predict.py:        "--gpu_allow_growth",
deepreg/predict.py:        help="Prevent TensorFlow from reserving all available GPU memory",
deepreg/predict.py:        gpu=args.gpu,
deepreg/predict.py:        gpu_allow_growth=args.gpu_allow_growth,
deepreg/train.py:    gpu: str,
deepreg/train.py:    gpu_allow_growth: bool = True,
deepreg/train.py:    :param gpu: which local gpu to use to train.
deepreg/train.py:    :param gpu_allow_growth: whether to allocate whole GPU memory for training.
deepreg/train.py:    if gpu is not None:
deepreg/train.py:        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
deepreg/train.py:        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = (
deepreg/train.py:            "true" if gpu_allow_growth else "false"
deepreg/train.py:    # use strategy to support multiple GPUs
deepreg/train.py:    # the network is mirrored in each GPU so that we can use larger batch size
deepreg/train.py:    num_devices = max(len(tf.config.list_physical_devices("GPU")), 1)
deepreg/train.py:        "--gpu",
deepreg/train.py:        help="GPU index for training."
deepreg/train.py:        "-g for using GPU remotely"
deepreg/train.py:        '-g "0" for using GPU 0'
deepreg/train.py:        '-g "0,1" for using GPU 0 and 1.',
deepreg/train.py:        "--gpu_allow_growth",
deepreg/train.py:        help="Prevent TensorFlow from reserving all available GPU memory",
deepreg/train.py:        gpu=args.gpu,
deepreg/train.py:        gpu_allow_growth=args.gpu_allow_growth,
examples/custom_backbone.py:    gpu="",
examples/custom_backbone.py:    gpu_allow_growth=True,
examples/custom_parameterized_image_label_loss.py:    gpu="",
examples/custom_parameterized_image_label_loss.py:    gpu_allow_growth=True,
examples/custom_image_label_loss.py:    gpu="",
examples/custom_image_label_loss.py:    gpu_allow_growth=True,
demos/unpaired_ct_lung/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/unpaired_ct_lung/demo_train.py:    "deepreg_train --gpu '0' "
demos/unpaired_ct_lung/demo_train.py:    gpu="0",
demos/unpaired_ct_lung/demo_train.py:    gpu_allow_growth=True,
demos/unpaired_ct_lung/demo_predict.py:    "deepreg_predict --gpu '' "
demos/unpaired_ct_lung/demo_predict.py:    gpu="0",
demos/unpaired_ct_lung/demo_predict.py:    gpu_allow_growth=True,
demos/grouped_mr_heart/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/grouped_mr_heart/demo_train.py:    "deepreg_train --gpu '0' "
demos/grouped_mr_heart/demo_train.py:    gpu="0",
demos/grouped_mr_heart/demo_train.py:    gpu_allow_growth=True,
demos/grouped_mr_heart/demo_predict.py:    "deepreg_predict --gpu '' "
demos/grouped_mr_heart/demo_predict.py:    gpu="0",
demos/grouped_mr_heart/demo_predict.py:    gpu_allow_growth=True,
demos/paired_mrus_brain/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/paired_mrus_brain/demo_train.py:    "deepreg_train --gpu '0' "
demos/paired_mrus_brain/demo_train.py:    gpu="0",
demos/paired_mrus_brain/demo_train.py:    gpu_allow_growth=True,
demos/paired_mrus_brain/demo_predict.py:    "deepreg_predict --gpu '' "
demos/paired_mrus_brain/demo_predict.py:    gpu="0",
demos/paired_mrus_brain/demo_predict.py:    gpu_allow_growth=True,
demos/paired_mrus_prostate/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/paired_mrus_prostate/demo_train.py:    "deepreg_train --gpu '0' "
demos/paired_mrus_prostate/demo_train.py:    gpu="0",
demos/paired_mrus_prostate/demo_train.py:    gpu_allow_growth=True,
demos/paired_mrus_prostate/demo_predict.py:    "deepreg_predict --gpu '' "
demos/paired_mrus_prostate/demo_predict.py:    gpu="0",
demos/paired_mrus_prostate/demo_predict.py:    gpu_allow_growth=True,
demos/grouped_mask_prostate_longitudinal/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/grouped_mask_prostate_longitudinal/demo_train.py:    "deepreg_train --gpu '0' "
demos/grouped_mask_prostate_longitudinal/demo_train.py:    gpu="0",
demos/grouped_mask_prostate_longitudinal/demo_train.py:    gpu_allow_growth=True,
demos/grouped_mask_prostate_longitudinal/demo_predict.py:    "deepreg_predict --gpu '0' "
demos/grouped_mask_prostate_longitudinal/demo_predict.py:    gpu="0",
demos/grouped_mask_prostate_longitudinal/demo_predict.py:    gpu_allow_growth=True,
demos/unpaired_mr_brain/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/unpaired_mr_brain/demo_train.py:    "deepreg_train --gpu '0' "
demos/unpaired_mr_brain/demo_train.py:    gpu="0",
demos/unpaired_mr_brain/demo_train.py:    gpu_allow_growth=True,
demos/unpaired_mr_brain/demo_predict.py:    "deepreg_predict --gpu '' "
demos/unpaired_mr_brain/demo_predict.py:    gpu="0",
demos/unpaired_mr_brain/demo_predict.py:    gpu_allow_growth=True,
demos/unpaired_ct_abdomen/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/unpaired_ct_abdomen/demo_train.py:    "deepreg_train --gpu '0' "
demos/unpaired_ct_abdomen/demo_train.py:    gpu="0",
demos/unpaired_ct_abdomen/demo_train.py:    gpu_allow_growth=True,
demos/unpaired_ct_abdomen/demo_predict.py:    "deepreg_predict --gpu '' "
demos/unpaired_ct_abdomen/demo_predict.py:    gpu="0",
demos/unpaired_ct_abdomen/demo_predict.py:    gpu_allow_growth=True,
demos/unpaired_us_prostate_cv/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/unpaired_us_prostate_cv/demo_train.py:    "deepreg_train --gpu '0' "
demos/unpaired_us_prostate_cv/demo_train.py:    gpu="0",
demos/unpaired_us_prostate_cv/demo_train.py:    gpu_allow_growth=True,
demos/unpaired_us_prostate_cv/demo_predict.py:    "deepreg_predict --gpu '' "
demos/unpaired_us_prostate_cv/demo_predict.py:    gpu="0",
demos/unpaired_us_prostate_cv/demo_predict.py:    gpu_allow_growth=True,
demos/paired_ct_lung/README.md:Here the training is launched using the GPU of index 0 with a limited number of steps
demos/paired_ct_lung/demo_train.py:    "deepreg_train --gpu '0' "
demos/paired_ct_lung/demo_train.py:    gpu="0",
demos/paired_ct_lung/demo_train.py:    gpu_allow_growth=True,
demos/paired_ct_lung/demo_predict.py:    "deepreg_predict --gpu '' "
demos/paired_ct_lung/demo_predict.py:    gpu="0",
demos/paired_ct_lung/demo_predict.py:    gpu_allow_growth=True,

```
