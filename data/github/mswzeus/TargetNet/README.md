# https://github.com/mswzeus/TargetNet

```console
evaluate_model.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TargetNet.yaml:  - cudatoolkit=11.3.1=h2bc3f7f_2
TargetNet.yaml:  - pytorch=1.10.2=py3.8_cuda11.3_cudnn8.2.0_0
TargetNet.yaml:  - pytorch-mutex=1.0=cuda
README.md:conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge
README.md:CUDA_VISIBLE_DEVICES=0 python train_model.py --data-config config/data/miRAW_train.json --model-config config/model/TargetNet.json --run-config config/run/run.json --output-path results/TargetNet_training/
README.md:CUDA_VISIBLE_DEVICES=0 python evaluate_model.py --data-config config/data/miRAW_eval.json --model-config config/model/TargetNet.json --run-config config/run/run.json --checkpoint pretrained_models/TargetNet.pt --output-path results/TargetNet-evaluation/
train_model.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src/config.py:    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
src/train.py:        # set gpu configurations
src/utils.py:    torch.cuda.manual_seed(seed)
src/utils.py:    torch.cuda.manual_seed_all(seed)

```
