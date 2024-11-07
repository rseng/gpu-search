# https://github.com/geoelements/gns

```console
slurm_scripts/parallel.sh:#SBATCH -p gpu-a100          # Queue (partition) name
slurm_scripts/train.sh:#SBATCH -p gpu-a100              # Queue (partition) name
slurm_scripts/train.sh:--cuda_device_number=0 \
slurm_scripts/train_parallel_multinode.sh:#SBATCH -p gpu-a100              # Queue (partition) name
slurm_scripts/train_gns_parallel.sh:GPU_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
slurm_scripts/train_gns_parallel.sh:LAUNCHER+="--nnodes=$NNODES  --nproc_per_node=$GPU_PER_NODE \
slurm_scripts/train_both.sh:#SBATCH -p gpu-a100               # Queue (partition) name
slurm_scripts/train_both.sh:--cuda_device_number=0 &
slurm_scripts/train_both.sh:#--cuda_device_number=0 &
slurm_scripts/train_both.sh:--cuda_device_number=1
slurm_scripts/train_both.sh:#--cuda_device_number=1
module.sh:ml cuda/12.0
module.sh:ml nccl
docs/README.md:Graph Network-based Simulator (GNS) is a framework for developing generalizable, efficient, and accurate machine learning (ML)-based surrogate models for particulate and fluid systems using Graph Neural Networks (GNNs). GNS code is a viable surrogate for numerical methods such as Material Point Method, Smooth Particle Hydrodynamics and Computational Fluid dynamics. GNS exploits distributed data parallelism to achieve fast multi-GPU training. The GNS code can handle complex boundary conditions and multi-material interactions.  GNS is a viable surrogate for numerical models such as Material Point Method, Smooth Particle Hydrodynamics and Computational Fluid dynamics.
test/test_pytorch_cuda_gpu.py:print(torch.cuda.is_available())
test/test_pytorch_cuda_gpu.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
README.md:Graph Network-based Simulator (GNS) is a generalizable, efficient, and accurate machine learning (ML)-based surrogate simulator for particulate and fluid systems using Graph Neural Networks (GNNs). GNS code is a viable surrogate for numerical methods such as Material Point Method, Smooth Particle Hydrodynamics and Computational Fluid dynamics. GNS exploits distributed data parallelism to achieve fast multi-GPU training. The GNS code can handle complex boundary conditions and multi-material interactions.
README.md:**cuda_device_number (Integer)** 
README.md:Base CUDA device (zero indexed).
README.md:Default is None so default CUDA device will be used.
README.md:**n_gpus (Integer)** 
README.md:Number of GPUs to use for training.
README.md:**cuda_device_number (Integer)**
README.md:Allows specifying a particular CUDA device for training or evaluation, enabling the use of specific GPUs in multi-GPU setups.
README.md:GNS uses [pytorch geometric](https://www.pyg.org/) and [CUDA](https://developer.nvidia.com/cuda-downloads). These packages have specific requirements, please see [PyG installation]((https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for details. 
README.md:GNS can be trained in parallel on multiple nodes with multiple GPUs.
README.md:> GNS scaling results on [TACC Frontera GPU nodes](https://docs.tacc.utexas.edu/hpc/frontera/#table3) with RTX-5000 GPUs.
README.md:> GNS scaling result on [TACC lonestar6 GPU nodes](https://docs.tacc.utexas.edu/hpc/lonestar6/#table2) with A100 GPUs.
README.md:#### Single Node, Multi-GPU
README.md:python -m torch.distributed.launch --nnodes=1  --nproc_per_node=[GPU_PER_NODE] --node_rank=[LOCAL_RANK] --master_addr=[MAIN_RANK] gns/train_multinode.py [ARGS] 
README.md:#### Multi-node, Multi-GPU
README.md:python -m torch.distributed.launch --nnodes=[NNODES]  --nproc_per_node=[GPU_PER_NODE] --node_rank=[LOCAL_RANK] --master_addr=[MAIN_RANK ]gns/train_multinode.py [ARGS] 
build_venv_frontera.sh:module load cuda/12
build_venv_frontera.sh:  echo 'test_pytorch_cuda_gpu.py -> True if GPU'
build_venv_frontera.sh:  python3 test/test_pytorch_cuda_gpu.py
build_venv.sh:echo 'test_pytorch_cuda_gpu.py -> True if GPU'
build_venv.sh:python test/test_pytorch_cuda_gpu.py
gns/learned_simulator.py:      device: Runtime device (cuda or cpu).
gns/train_multinode.py:flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
gns/train_multinode.py:  serial_simulator = serial_simulator.to('cuda')
gns/train_multinode.py:    device: cuda device local rank
gns/train_multinode.py:  serial_simulator = serial_simulator.to('cuda')
gns/train_multinode.py:        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
gns/train_multinode.py:  if torch.cuda.is_available():
gns/train_multinode.py:    device: PyTorch device 'cpu' or 'cuda'.
gns/train_multinode.py:  device = torch.device('cuda')
gns/train_multinode.py:      backend="nccl",
gns/train_multinode.py:  # instead of torch.cuda.device_count(), we use dist.get_world_size() to get world size
gns/train_multinode.py:  torch.cuda.set_device(local_rank)
gns/train_multinode.py:  torch.cuda.manual_seed(0)
gns/train_multinode.py:    # world_size = torch.cuda.device_count()
gns/train_multinode.py:    # if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
gns/train_multinode.py:    #   device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
gns/train.py:flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
gns/train.py:flags.DEFINE_integer("n_gpus", 1, help="The number of GPUs to utilize for training.")
gns/train.py:  if device == torch.device("cuda"):
gns/train.py:  if device == torch.device("cuda"):
gns/train.py:      if device == torch.device("cuda"):
gns/train.py:        simulator.module.parameters() if device == torch.device("cuda") else simulator.parameters())
gns/train.py:    if device == torch.device("cuda")
gns/train.py:  print(f"rank = {rank}, cuda = {torch.cuda.is_available()}")
gns/train.py:      if device == torch.device("cuda"):
gns/train.py:        device_or_rank = rank if device == torch.device("cuda") else device
gns/train.py:        pred_acc, target_acc = (simulator.module.predict_accelerations if device == torch.device("cuda") else simulator.predict_accelerations)(
gns/train.py:      if device == torch.device("cuda"):
gns/train.py:        if device == torch.device("cuda"):
gns/train.py:  if torch.cuda.is_available():
gns/train.py:    device: PyTorch device 'cpu' or 'cuda'.
gns/train.py:  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gns/train.py:  if device == torch.device('cuda'):
gns/train.py:    # Train on gpu 
gns/train.py:    if device == torch.device('cuda'):
gns/train.py:      available_gpus = torch.cuda.device_count()
gns/train.py:      print(f"Available GPUs = {available_gpus}")
gns/train.py:      # Set the number of GPUs based on availability and the specified number
gns/train.py:      if FLAGS.n_gpus is None or FLAGS.n_gpus > available_gpus:
gns/train.py:        world_size = available_gpus
gns/train.py:        if FLAGS.n_gpus is not None:
gns/train.py:          print(f"Warning: The number of GPUs specified ({FLAGS.n_gpus}) exceeds the available GPUs ({available_gpus})")
gns/train.py:        world_size = FLAGS.n_gpus
gns/train.py:      # Print the status of GPU usage
gns/train.py:      print(f"Using {world_size}/{available_gpus} GPUs")
gns/train.py:      # Spawn training to GPUs
gns/train.py:    world_size = torch.cuda.device_count()
gns/train.py:    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
gns/train.py:      device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
gns/distribute.py:    torch.distributed.init_process_group(backend="nccl",
example/inverse_problem/inverse.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_venv.sh:# echo 'test_pytorch_cuda_gpu.py -> True if GPU'
start_venv.sh:# python test/test_pytorch_cuda_gpu.py
paper.md:The GNS implementation uses semi-implicit Euler integration to update the state of the particles based on the nodes predicted accelerations.  We introduce physics-inspired simple inductive biases, such as an inertial frame that allows learning algorithms to prioritize one solution over another, instead of learning to predict the inertial motion, the neural network learns to trivially predict a correction to the inertial trajectory, reducing learning time.  We developed an open-source, PyTorch-based GNS that predicts the dynamics of fluid and particulate systems [@Kumar_Graph_Network_Simulator_2022].  GNS trained on trajectory data is generalizable to predict particle kinematics in complex boundary conditions not seen during training.  \autoref{fig:gns-mpm} shows the GNS prediction of granular flow around complex obstacles trained on 20 million steps with 40 trajectories on NVIDIA A100 GPUs.  The trained model accurately predicts within 5\% error of its associated material point method (MPM) simulation.  The predictions are 5,000x faster than traditional MPM simulations (2.5 hours for MPM simulations versus 20 s for GNS simulation of granular flow) and are widely used for solving optimization, control and inverse-type problems.  In addition to surrogate modeling, GNS trained on flow problems is also used as an oracle to predict the dynamics of flows to identify critical regions of interest for in situ rendering and visualization [@kumar2022insitu].  The GNS code is distributed under the open-source MIT license and is available on [GitHub Geoelements GNS](https://github.com/geoelements/gns).
paper.md:@sanchez2020learning developed a reference GNS implementation based on TensorFlow v1 [@tensorflow2015whitepaper].  Although the reference implementation runs both on CPU and GPU, it doesn't achieve multi-GPU scaling.  Furthermore, the dependence on TensorFlow v1 limits its ability to leverage features such as eager execution in TF v2.  We develop a scalable and modular GNS using PyTorch using the Distributed Data Parallel model to run on multi-GPU systems.
paper.md:- CPU and GPU training
paper.md:- Parallel training on multi-GPUs
paper.md:The GNS is parallelized to run across multiple GPUs using the PyTorch Distributed Data Parallel (DDP) model.  The DDP model spawns as many GNS models as the number of GPUs, distributing the dataset across all GPU nodes.  Consider, our training dataset with 20 simulations, each with 206 time steps of positional data $x_i$, which yields $(206 - 6) \times 20 = 4000$ training trajectories.  We subtract six position from the GNS training dataset as we utilize five previous velocities, computed from six positions, to predict the next position.  The 4000 training tajectories are subsequently distributed equally to the four GPUs (1000 training trajectories/GPU).  Assuming a batch size of 2, each GPU handles 500 trajectories in a batch.  The loss from the training trajectories are computed as difference between accelerations of GNS prediction and actual trajectories. 
paper.md:where $n$ is the number of particles (nodes) and $\theta$ is the learnable parameter in the GNS. In DDP, the gradient $\nabla (f(\theta))$ is computed as the average gradient across all GPUs as shown in \autoref{fig:gns-ddp}.
paper.md:We tested the strong scaling of the GNS code on a single node of Lonestar 6 at the Texas Advanced Computing Center equipped with three NVIDIA A100 GPUs.  We evaluated strong scaling for the [WaterDropSample](https://www.designsafe-ci.org/data/browser/public/designsafe.storage.published//PRJ-3702/WaterDropSample) dataset for 6000 training steps using the recommended `nccl` DDP backend.  \autoref{fig:gns-scaling} shows linear strong scaling performance.
paper.md:![GNS strong-scaling on up to three NVIDIA A100 GPUs.\label{fig:gns-scaling}](figs/gns-scaling.png)
meshnet/learned_simulator.py:          device: Runtime device (cuda or cpu).
meshnet/train.py:flags.DEFINE_integer("cuda_device_number", None, help="CUDA device (zero indexed), default is None so default CUDA device will be used.")
meshnet/train.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meshnet/train.py:    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
meshnet/train.py:    if FLAGS.cuda_device_number is not None and torch.cuda.is_available():
meshnet/train.py:        device = torch.device(f'cuda:{int(FLAGS.cuda_device_number)}')
meshnet/normalization.py:    def __init__(self, size, max_accumulations=10**6, std_epsilon=1e-8, name='Normalizer', device='cuda'):

```
