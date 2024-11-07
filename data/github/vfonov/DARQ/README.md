# https://github.com/vfonov/DARQ

```console
python/model/resnet_qc.py:    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
python/pytorch_171_explicit.txt:https://repo.anaconda.com/pkgs/main/linux-64/cudatoolkit-11.0.221-h6bb024c_0.conda
python/pytorch_171_explicit.txt:https://conda.anaconda.org/pytorch/linux-64/pytorch-1.7.1-py3.8_cuda11.0.221_cudnn8.0.5_0.tar.bz2
python/aqc_training.py:            inputs = v_sample_batched['image' ].cuda()
python/aqc_training.py:            labels = v_sample_batched['status'].cuda()
python/aqc_training.py:                dist = v_sample_batched['dist'].float().cuda()
python/aqc_training.py:    model = model.cuda()
python/aqc_training.py:            inputs = sample_batched['image'].cuda()
python/aqc_training.py:                dist = sample_batched['dist'].float().cuda()
python/aqc_training.py:                labels = sample_batched['status'].cuda()
python/start_tensorboard.sh:#export CUDA_VISIBLE_DEVICES=None
python/aqc_apply.py:    parser.add_argument('--gpu', action="store_true", default=False,
python/aqc_apply.py:                        help='Run inference in gpu')
python/aqc_apply.py:    if params.gpu:
python/aqc_apply.py:        model=model.cuda()
python/aqc_apply.py:                if params.gpu: inputs = inputs.cuda()
python/aqc_apply.py:                if params.gpu: outputs = outputs.cpu()
README.md:* GPU version
README.md:    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=<your cuda version>  -c pytorch 
README.md:    * `aqc_convert_to_cpu.py`- helper script to convert network from GPU to CPU
python-tf/aqc_training.py:    "multigpu", default=False,
python-tf/aqc_training.py:    help="Use all available GPUs")
python-tf/aqc_training.py:    "gpu", default=False,
python-tf/aqc_training.py:    help="Train on GPU using Estimator")
python-tf/start_tensorboard.sh:#export CUDA_VISIBLE_DEVICES=None
python-tf/estimator.py:        'multigpu':flags.multigpu,
python-tf/estimator.py:        'gpu':flags.gpu,
python-tf/estimator.py:    if flags.multigpu or flags.gpu:
python-tf/estimator.py:    if training_active and not params["multigpu"] and not params["gpu"]:
python-tf/estimator.py:    if flags.multigpu: # train on multiple GPUs
python-tf/estimator.py:    elif flags.gpu: # train on single GPU using Estimator

```
