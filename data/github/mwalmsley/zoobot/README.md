# https://github.com/mwalmsley/zoobot

```console
setup.py:        "Environment :: GPU :: NVIDIA CUDA"
setup.py:            # A100 GPU currently only seems to support cuda 11.3 on manchester cluster, let's stick with this version for now
setup.py:            # very latest version wants cuda 11.6
setup.py:        # for GPU, you will also need e.g. cudatoolkit=11.3, 11.6
setup.py:        ],  # exactly as above, but _cu121 for cuda 12.1 (the current default)
setup.py:            'tensorflow == 2.10.0',  # 2.11.0 turns on XLA somewhere which then fails on multi-GPU...TODO
setup.py:        # for GPU, you will also need cudatoolkit=11.2 and cudnn=8.1.0 (note - 11.3 NOT supported by TF)
Dockerfile:FROM nvidia/cuda:11.3.1-base-ubuntu20.04
Dockerfile:# container already has CUDA 11.3
docs/pretrained_models.rst:ConvNeXT-Nano performs very well while still being small enough to train on a single gaming GPU.
docs/pretrained_models.rst:All these models are much larger and need cluster-grade GPUs (e.g. V100 or above).
docs/guides/how_the_code_fits_together.rst:PyTorch requires a lot of boilerplate code to train models, especially at scale (e.g. multi-node, multi-GPU).
docs/guides/how_the_code_fits_together.rst:LightningModules can be passed to a Lightning ``Trainer`` object. This handles running the training in practice (e.g. how to distribute training onto a GPU, how many epochs to run, etc.).
zoobot/pytorch/datasets/webdatamodule.py:        # for test/predict, always single GPU anyway
zoobot/pytorch/predictions/predict_on_catalog.py:        **trainer_kwargs  # e.g. gpus
zoobot/pytorch/estimators/define_model.py:    # Would use lambda but multi-gpu doesn't support as lambda can't be pickled
zoobot/pytorch/estimators/cuda_check.py:print('CUDA', torch.cuda.is_available())
zoobot/pytorch/estimators/cuda_check.py:print('CUDA version', torch.version.cuda)
zoobot/pytorch/training/train_with_pytorch_lightning.py:    gpus=2,
zoobot/pytorch/training/train_with_pytorch_lightning.py:        gpus (int, optional): Multi-GPU training. Uses distributed data parallel - essentially, full dataset is split by GPU. See torch docs. Defaults to 2.
zoobot/pytorch/training/train_with_pytorch_lightning.py:            pass # another gpu process may have just made it
zoobot/pytorch/training/train_with_pytorch_lightning.py:    if (gpus is not None) and (gpus > 1):
zoobot/pytorch/training/train_with_pytorch_lightning.py:        logging.info('Using multi-gpu training')
zoobot/pytorch/training/train_with_pytorch_lightning.py:    if gpus > 0:
zoobot/pytorch/training/train_with_pytorch_lightning.py:        accelerator = 'gpu'
zoobot/pytorch/training/train_with_pytorch_lightning.py:        devices = gpus
zoobot/pytorch/training/train_with_pytorch_lightning.py:    if (gpus is not None) and (num_workers * gpus > os.cpu_count()):
zoobot/pytorch/training/train_with_pytorch_lightning.py:            """num_workers * gpu > num cpu.
zoobot/pytorch/training/train_with_pytorch_lightning.py:            'gpus': gpus,
zoobot/pytorch/training/train_with_pytorch_lightning.py:        logging.info(f'Testing on {checkpoint_callback.best_model_path} with single GPU. Be careful not to overfit your choices to the test data...')
zoobot/pytorch/training/train_with_pytorch_lightning.py:            ckpt_path=checkpoint_callback.best_model_path  # can optionally point to a specific checkpoint here e.g. "/share/nas2/walml/repos/gz-decals-classifiers/results/early_stopping_1xgpu_greyscale/checkpoints/epoch=26-step=16847.ckpt"
zoobot/pytorch/training/train_with_pytorch_lightning.py:    # # you can do this to see images, but if you do, wandb will cause training to silently hang before starting if you do this on multi-GPU runs
zoobot/pytorch/training/train_with_pytorch_lightning.py:    # may help stop tasks getting left on gpu after slurm exit?
zoobot/pytorch/training/finetune.py:        )  # trained on GPU, may need map_location='cpu' if you get a device error
zoobot/pytorch/training/finetune.py:    if torch.cuda.is_available():
zoobot/pytorch/training/finetune.py:        # necessary to load gpu-trained model on cpu
zoobot/pytorch/training/finetune.py:        devices (str, optional): number of devices for training (typically, num. GPUs). Defaults to 'auto'.
zoobot/pytorch/training/finetune.py:        accelerator (str, optional): which device to use (typically 'gpu' or 'cpu'). Defaults to 'auto'.
zoobot/pytorch/examples/finetuning/finetune_binary_classification.py:      trainer_kwargs={'accelerator': 'gpu'}  
zoobot/pytorch/examples/finetuning/finetune_counts_full_tree.py:        accelerator = 'gpu'
zoobot/pytorch/examples/train_from_scratch/minimal_example.py:    gpus = 1
zoobot/pytorch/examples/train_from_scratch/minimal_example.py:        gpus=gpus,
zoobot/pytorch/examples/train_from_scratch/train_model_on_catalog.py:    parser.add_argument('--accelerator', type=str, default='gpu')
zoobot/pytorch/examples/train_from_scratch/train_model_on_catalog.py:    parser.add_argument('--gpus', default=1, type=int)
zoobot/pytorch/examples/train_from_scratch/train_model_on_catalog.py:                        help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
zoobot/pytorch/examples/train_from_scratch/train_model_on_catalog.py:        # https://docs.wandb.ai/guides/integrations/lightning#how-to-use-multiple-gpus-with-lightning-and-w-and-b
zoobot/pytorch/examples/train_from_scratch/train_model_on_catalog.py:        gpus=args.gpus,
zoobot/pytorch/examples/representations/get_representations.py:    accelerator = 'cpu'  # or 'gpu' if available
zoobot/tensorflow/training/train_with_keras.py:    gpus=2,
zoobot/tensorflow/training/train_with_keras.py:        gpus (int, optional): Num. gpus to use. Defaults to 2.
zoobot/tensorflow/training/train_with_keras.py:    if gpus > 1:
zoobot/tensorflow/training/train_with_keras.py:        strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
zoobot/tensorflow/training/train_with_keras.py:        # each GPU will calculate loss (hence gradients) for that device's sub-batch
zoobot/tensorflow/training/train_with_keras.py:        assert strategy.num_replicas_in_sync == gpus
zoobot/tensorflow/training/train_with_keras.py:        logging.info('Using single or no GPU, not distributed')
zoobot/tensorflow/training/train_with_keras.py:        def loss(x, y): return tf.reduce_sum(multiquestion_loss(x, y)) / (batch_size/gpus)        
zoobot/tensorflow/training/train_with_keras.py:        # be careful to define this within the context_manager, so it is also mirrored if on multi-gpu
zoobot/tensorflow/training/train_with_keras.py:            # this currently only works on 1 GPU - see Keras issue
zoobot/tensorflow/training/train_with_keras.py:            'gpus': gpus,
zoobot/tensorflow/training/train_with_keras.py:        jit_compile=False  # don't use XLA, it fails on multi-GPU. Might consider on one GPU.
zoobot/tensorflow/examples/make_predictions_loop.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/make_predictions_loop.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/make_predictions_loop.py:    if gpus:
zoobot/tensorflow/examples/make_predictions_loop.py:        for gpu in gpus:
zoobot/tensorflow/examples/make_predictions_loop.py:          tf.config.experimental.set_memory_growth(gpu, True)
zoobot/tensorflow/examples/make_predictions_loop.py:    batch_size = 256  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100, 512 for 2xA100
zoobot/tensorflow/examples/make_predictions.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/make_predictions.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/make_predictions.py:    if gpus:
zoobot/tensorflow/examples/make_predictions.py:        for gpu in gpus:
zoobot/tensorflow/examples/make_predictions.py:          tf.config.experimental.set_memory_growth(gpu, True)
zoobot/tensorflow/examples/make_predictions.py:    batch_size = 256  # 128 for paper, you'll need a very good GPU. 8 for debugging, 64 for RTX 2070, 256 for A100
zoobot/tensorflow/examples/finetuning/finetune_advanced.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/finetuning/finetune_advanced.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/finetuning/finetune_advanced.py:    if gpus:
zoobot/tensorflow/examples/finetuning/finetune_advanced.py:        for gpu in gpus:
zoobot/tensorflow/examples/finetuning/finetune_advanced.py:          tf.config.experimental.set_memory_growth(gpu, True)
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    if gpus:
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:        for gpu in gpus:
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:            tf.config.experimental.set_memory_growth(gpu, True)
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    # check which GPU we're using
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    physical_devices = tf.config.list_physical_devices('GPU')
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    logging.info('GPUs: {}'.format(physical_devices))
zoobot/tensorflow/examples/deprecated/train_model_on_shards.py:    parser.add_argument('--gpus', default=1, type=int)
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    if gpus:
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:        for gpu in gpus:
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:            tf.config.experimental.set_memory_growth(gpu, True)
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    # check which GPU we're using
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    physical_devices = tf.config.list_physical_devices('GPU')
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    logging.info('GPUs: {}'.format(physical_devices))
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:    parser.add_argument('--gpus', default=1, type=int)
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:                        help='If true, use automatic mixed precision (via PyTorch Lightning) to reduce GPU memory use (~x2). Else, use full (32 bit) precision')
zoobot/tensorflow/examples/train_from_scratch/train_model_on_catalog.py:        gpus=args.gpus,
zoobot/tensorflow/examples/finetune_on_fixed_representation.py:    # useful to avoid errors on small GPU
zoobot/tensorflow/examples/finetune_on_fixed_representation.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
zoobot/tensorflow/examples/finetune_on_fixed_representation.py:    if gpus:
zoobot/tensorflow/examples/finetune_on_fixed_representation.py:        for gpu in gpus:
zoobot/tensorflow/examples/finetune_on_fixed_representation.py:          tf.config.experimental.set_memory_growth(gpu, True)
tests/pytorch/test_train_with_pytorch_lightning.py:        gpus=0,
only_for_me/slurm_scripts/train_pytorch_model.sh:nvidia-smi
only_for_me/slurm_scripts/train_pytorch_model.sh:# $PYTHON /share/nas2/walml/repos/zoobot/zoobot/pytorch/estimators/cuda_check.py \
only_for_me/slurm_scripts/train_pytorch_model.sh:    --gpus 1  \
only_for_me/slurm_scripts/finetune_advanced.sh:nvidia-smi
only_for_me/slurm_scripts/finetune_advanced.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64
only_for_me/slurm_scripts/train_pytorch_model_manual_cluster.sh:nvidia-smi
only_for_me/slurm_scripts/train_pytorch_model_manual_cluster.sh:export NCCL_DEBUG=INFO
only_for_me/slurm_scripts/train_pytorch_model_manual_cluster.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64
only_for_me/slurm_scripts/train_pytorch_model_manual_cluster.sh:# with DDP, batch size is per gpu
only_for_me/slurm_scripts/finetune_counts_full_tree.sh:nvidia-smi
only_for_me/slurm_scripts/train_pytorch_model_multinode.sh:nvidia-smi
only_for_me/slurm_scripts/train_pytorch_model_multinode.sh:export NCCL_DEBUG=INFO
only_for_me/slurm_scripts/train_pytorch_model_multinode.sh:export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64
only_for_me/slurm_scripts/train_pytorch_model_multinode.sh:# with DDP, batch size is per gpu
only_for_me/load_resnet_into_detectron2.py:    cfg.MODEL.DEVICE='cpu'  # model was trained on gpu, but you might not have one - simplest to load on cpu
only_for_me/load_resnet_into_detectron2.py:    ckpt_to_load = '/home/walml/repos/resnet_det_1gpu_b256/checkpoints/epoch=42-step=44032.ckpt'
README.md:You can retrain Zoobot in the cloud with a free GPU using this [Google Colab notebook](https://colab.research.google.com/drive/1A_-M3Sz5maQmyfW2A7rEu-g_Zi0RMGz5?usp=sharing). To install locally, keep reading.
README.md:    # Zoobot with PyTorch and a GPU. Requires CUDA 12.1 (or CUDA 11.8, if you use `_cu118` instead)
README.md:    # OR Zoobot with PyTorch and no GPU
README.md:To use a GPU, you must *already* have CUDA installed and matching the versions above.
README.md:I share my install steps [here](#install_cuda). GPUs are optional - Zoobot will run retrain fine on CPU, just slower.
README.md:### (Optional) Install PyTorch with CUDA
README.md:<a name="install_cuda"></a>
README.md:*If you're not using a GPU, skip this step. Use the pytorch-cpu option in the section below.*
README.md:Install PyTorch 2.1.0 and compatible CUDA drivers. I highly recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to do this. Conda will handle both creating a new virtual environment (`conda create`) and installing CUDA (`cudatoolkit`, `cudnn`)
README.md:CUDA 12.1 for PyTorch 2.1.0:
README.md:    conda install -c conda-forge cudatoolkit=12.1
paper/paper.md: - name: Physical Research Laboratory, Navrangpura, Ahmedabad, India
paper/paper.md:The API abstracts away engineering details such as efficiently loading astronomical images, multi-GPU training, iteratively finetuning deeper model layers, and so forth.
docker-compose.yml:    image: zoobot:cuda
benchmarks/comparison_debugging.py:Both use 2 GPUs
benchmarks/comparison_debugging.py:Do the losses agree on 1 GPU mode? This will tell if the difference is from per-replica aggregation, or from e.g. averaging over all values not all rows
benchmarks/comparison_debugging.py:- PT has no 2x factor change in loss with gpu number (good, this is intuitive)
benchmarks/comparison_debugging.py:- TF seems to increase by a factor of 2 (11 to 20) when going from 2 to 1 GPUs. Aka, loss is divided by num. GPUs. Likely because batch is divided across gpus? Add explicit GPU factor and restart
benchmarks/comparison_debugging.py:- TF is sum of multi-q loss / batch size, divided by num GPUs
benchmarks/comparison_debugging.py:- PT was mean of multi-q loss (aka average across questions, not just sum), independent of num GPUs.
benchmarks/comparison_debugging.py:- TF to be multiplied by num gpus in train_with_keras loss func, counteracting current num gpus divisor (x2 factor, usually)
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_debug_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED,DEBUG_STRING='--debug' $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# minimal hardware - 1 gpu, no mixed precision
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_min_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:sbatch --job-name=evo_py_gr_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:sbatch --job-name=evo_py_gr_eff_300_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:sbatch --job-name=evo_py_co_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:sbatch --job-name=evo_py_co_eff_300_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=128,RESIZE_AFTER_CROP=300,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_res18_224_$SEED --export=ARCHITECTURE=resnet18,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_res18_300_$SEED --export=ARCHITECTURE=resnet18,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_res50_224_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_res50_300_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=300,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_co_res50_224_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_vittiny_224_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_co_vittiny_224_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_vitsmall_224_$SEED --export=ARCHITECTURE=maxvit_small_224,BATCH_SIZE=64,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_$SEED --export=ARCHITECTURE=convnext_nano,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_vittinyp16_224_$SEED --export=ARCHITECTURE=vit_tiny_patch16_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_co_res50_224_fullprec_$SEED --export=ARCHITECTURE=resnet50,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_evo,COLOR_STRING=--color,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_128px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=128,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_c_128px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=128,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_gr_64px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=64,DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=evo_py_c_64px_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=64,DATASET=gz_evo,COLOR_STRING=--color,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_gr_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_co_eff_224_$SEED --export=ARCHITECTURE=efficientnet_b0,BATCH_SIZE=256,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,COLOR_STRING=--color,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_vit_$SEED --export=ARCHITECTURE=maxvit_tiny_224,BATCH_SIZE=128,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/run_benchmarks.sh:# sbatch --job-name=dr5_py_vit_$SEED --export=ARCHITECTURE=maxvit_small_224,BATCH_SIZE=64,RESIZE_AFTER_CROP=224,DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/pytorch/train_model_on_benchmark_dataset.py:    parser.add_argument('--gpus', dest='gpus', default=1, type=int)
benchmarks/pytorch/train_model_on_benchmark_dataset.py:        gpus=args.gpus,
benchmarks/pytorch/train_model_on_benchmark_dataset.py:        num_workers=11,  # system has 24 cpu, 12 cpu per gpu, leave a little wiggle room
benchmarks/pytorch/run_dataset_benchmark.sh:# GPUS=2
benchmarks/pytorch/run_dataset_benchmark.sh:nvidia-smi
benchmarks/pytorch/run_dataset_benchmark.sh:export NCCL_DEBUG=INFO
benchmarks/pytorch/run_dataset_benchmark.sh:    --gpus $GPUS \
benchmarks/pytorch/run_dataset_benchmark.sh:    --gpus $GPUS \
benchmarks/tensorflow/run_benchmarks.sh:# sbatch --job-name=dr5_tf_debug_$SEED --export=DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED,DEBUG_STRING='--debug' $TRAIN_JOB
benchmarks/tensorflow/run_benchmarks.sh:# minimal hardware - 1 gpu, no mixed precision
benchmarks/tensorflow/run_benchmarks.sh:# sbatch --job-name=dr5_tf_min_$SEED --export=DATASET=gz_decals_dr5,GPUS=1,SEED=$SEED $TRAIN_JOB
benchmarks/tensorflow/run_benchmarks.sh:# otherwise full hardware (standard setup) - 2 gpus, mixed precision
benchmarks/tensorflow/run_benchmarks.sh:sbatch --job-name=dr5_tf_gr_$SEED --export=DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/tensorflow/run_benchmarks.sh:# sbatch --job-name=dr5_tf_co_$SEED --export=DATASET=gz_decals_dr5,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,COLOR_STRING=--color,SEED=$SEED $TRAIN_JOB
benchmarks/tensorflow/run_benchmarks.sh:# sbatch --job-name=evo_tf_gr_$SEED --export=DATASET=gz_evo,MIXED_PRECISION_STRING=--mixed-precision,GPUS=2,SEED=$SEED $TRAIN_JOB
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    New features: add --distributed for multi-gpu, --wandb for weights&biases metric tracking, --color for color (does not perform better)
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    # useful to avoid errors on small GPU
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    if gpus:
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:        for gpu in gpus:
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:          tf.config.experimental.set_memory_growth(gpu, True)
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    # check which GPU we're using
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    physical_devices = tf.config.list_physical_devices('GPU') 
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:    logging.info('GPUs: {}'.format(physical_devices))
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:      strategy = tf.distribute.MirroredStrategy()  # one machine, one or more GPUs
benchmarks/tensorflow/deprecated/single_task_version_not_recommended/train_single_task_model.py:      logging.info('Using single GPU, not distributed')
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    # useful to avoid errors on small GPU
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    gpus = tf.config.experimental.list_physical_devices('GPU')
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    if gpus:
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:        for gpu in gpus:
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:            tf.config.experimental.set_memory_growth(gpu, True)
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    # check which GPU we're using
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    physical_devices = tf.config.list_physical_devices('GPU')
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    logging.info('GPUs: {}'.format(physical_devices))
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:    parser.add_argument('--gpus', default=2, type=int)
benchmarks/tensorflow/train_model_on_benchmark_dataset.py:        gpus=args.gpus,
benchmarks/tensorflow/run_dataset_benchmark.sh:# export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/share/apps/cudnn_8_1_0/cuda/lib64
benchmarks/tensorflow/run_dataset_benchmark.sh:nvidia-smi
benchmarks/tensorflow/run_dataset_benchmark.sh:BATCH_SIZE=512  # equivalent to 256 on PyTorch, with 2 GPUs
benchmarks/tensorflow/run_dataset_benchmark.sh:    --gpus $GPUS \
benchmarks/tensorflow/run_dataset_benchmark.sh:    --gpus $GPUS \

```
