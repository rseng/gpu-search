# https://github.com/astrockragh/Mangrove

```console
exp_example/guide.txt:'python run_experiment.py -f <folder> -gpu <1 if gpu, 0 if cpu>
exp_sweep_example/guide.txt:'python run_sweep.py -f <folder> -gpu <1 if gpu, 0 if cpu> -N <N_experiments>'
run_experiment.py:parser.add_argument("-gpu", "--gpu", type=str, required=False)
run_experiment.py:# parser.add_argument("-gpu", "--gpu", type=str, required=False)
run_experiment.py:if args.gpu=='1':
run_experiment.py:    from dev.train_script_gpu import train_model
README.md:Having restructured the merger tree, you can then do the training either as single experiments (using run_experiment.py) or as a sweep (using run_sweep.py). All the things required to do the training and tracking are in the dev folder. Here, you'll find loss functions, learning rate schedulers, models and a script for doing the training on the gpu/cpu (the cpu version is outdated).
README.md:`conda create --name jtorch pytorch==1.9.0 jupyter torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 matplotlib tensorboard --channel pytorch`
slurm_pysr/pysr.slurm:#SBATCH --gres=gpu:1            # number of gpus per node
slurm_pysr/pysr.slurm:python ../run_sweep.py -f exp_pysr -N 6 -gpu 1
slurm_pysr/pysr_free.slurm:#SBATCH --gres=gpu:1            # number of gpus per node
slurm_pysr/pysr_free.slurm:python ../run_sweep.py -f exp_pysr_free -N 6 -gpu 1
slurm_pysr/pysr_sparse.slurm:#SBATCH --gres=gpu:1            # number of gpus per node
slurm_pysr/pysr_sparse.slurm:python ../run_sweep.py -f exp_pysr_sparse -N 6 -gpu 1
dev/metrics.py:from torch.cuda import FloatTensor
dev/train_script_gpu.py:## Current GPU train loop              ##
dev/train_script_gpu.py:        sca=torch.cuda.FloatTensor(scales[targs])
dev/train_script_gpu.py:        ms=torch.cuda.FloatTensor(mus[targs])
dev/train_script_gpu.py:            er_loss = torch.cuda.FloatTensor([0])
dev/train_script_gpu.py:            si_loss = torch.cuda.FloatTensor([0])
dev/train_script_gpu.py:            rh_loss = torch.cuda.FloatTensor([0])
dev/loss_funcs.py:    '''This is written assuming normal pytorch, on cuda, inputs should be 4xbatchsize, 4xbatchsize, 4xbatchsize, 6xbatchsize'''
dev/loss_funcs.py:    A = zeros(N, N,bsize, device='cuda:0') # assuming you're on the gpu
run_sweep.py:parser.add_argument("-gpu", "--gpu", type=str, required=False)
run_sweep.py:# parser.add_argument("-gpu", "--gpu", type=str, required=False)
run_sweep.py:if args.gpu=='1':
run_sweep.py:    from dev.train_script_gpu import train_model

```
