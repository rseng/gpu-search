# https://github.com/chenwuperth/rgz_rcnn

```console
README.md:*ClaRAN* replaces the original RoI cropping layer with the [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) (STN) pooling to support a fully end-to-end training pipeline. An *unexpected* benefit of this is that the code also runs on laptops that may not have GPUs (with a much longer latency  of course --- e.g. 6 seconds compared to 100s of milliseconds per image).
README.md:This should compile a bunch of [Cython](https://cython.org/)/ C code (for bounding box calculation), and will produce the dynamic libraries under both CPUs and GPUs (if available).
README.md:#### To train your own RGZ model on GPU node managed by the SLURM job scheduler:
tools/demo.py:    parser.add_argument('--device_id', dest='device_id', help='device id to use for GPUs',
tools/demo.py:    if args.device.lower() == 'gpu':
tools/demo.py:        cfg.USE_GPU_NMS = True
tools/demo.py:        cfg.GPU_ID = args.device_id
tools/demo.py:        cfg.USE_GPU_NMS = False
tools/test_net.py:    if args.device == 'gpu':
tools/test_net.py:        cfg.USE_GPU_NMS = True
tools/test_net.py:        cfg.GPU_ID = args.device_id
tools/test_net.py:        cfg.USE_GPU_NMS = False
experiments/scripts/example_train_slurm.sh:#SBATCH --gres=gpu:1
experiments/scripts/example_train_slurm.sh:module load tensorflow/1.4.0-py27-gpu
experiments/scripts/example_train_slurm.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/example_train_slurm.sh:# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/example_train_slurm.sh:# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/example_train_slurm.sh:                    --device gpu \
experiments/scripts/example_train_slurm_pleiades.sh:#SBATCH --partition=mlgpu
experiments/scripts/example_train_slurm_pleiades.sh:#SBATCH --gres=gpu:1
experiments/scripts/example_train_slurm_pleiades.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/example_train_slurm_pleiades.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/example_train_slurm_pleiades.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/example_train_slurm_pleiades.sh:                    --device gpu \
experiments/scripts/plots/loss/test/slurm-42146.out:2018-09-19 08:42:52.392825: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42146.out:2018-09-19 08:42:52.393318: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42146.out:2018-09-19 08:42:52.393349: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42146.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_20k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-20000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42146.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42146.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42146.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42160.out:2018-09-19 11:58:06.358683: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42160.out:2018-09-19 11:58:06.359215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42160.out:2018-09-19 11:58:06.359246: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42160.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_80k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-80000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42160.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42160.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42160.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42158.out:2018-09-19 11:22:18.432338: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42158.out:2018-09-19 11:22:18.432936: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42158.out:2018-09-19 11:22:18.432971: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42158.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_60k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-60000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42158.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42158.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42158.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42163.out:2018-09-19 12:51:49.541187: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42163.out:2018-09-19 12:51:49.541843: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42163.out:2018-09-19 12:51:49.541874: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42163.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_30k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-30000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42163.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42163.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42163.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42153.out:2018-09-19 09:52:21.304195: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42153.out:2018-09-19 09:52:21.305037: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42153.out:2018-09-19 09:52:21.305079: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42153.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_10k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-10000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42153.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42153.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42153.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42168.out:2018-09-19 14:21:23.933138: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42168.out:2018-09-19 14:21:23.933719: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42168.out:2018-09-19 14:21:23.933749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42168.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_80k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-80000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42168.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42168.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42168.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42161.out:2018-09-19 12:16:05.084750: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42161.out:2018-09-19 12:16:05.085333: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42161.out:2018-09-19 12:16:05.085367: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42161.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_10k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-10000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42161.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42161.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42161.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42167.out:2018-09-19 14:03:29.928849: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42167.out:2018-09-19 14:03:29.929336: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42167.out:2018-09-19 14:03:29.929366: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42167.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_70k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-70000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42167.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42167.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42167.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42166.out:2018-09-19 13:45:34.873571: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42166.out:2018-09-19 13:45:34.874061: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42166.out:2018-09-19 13:45:34.874091: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42166.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_60k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-60000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42166.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42166.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42166.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42162.out:2018-09-19 12:33:54.802354: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42162.out:2018-09-19 12:33:54.802938: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42162.out:2018-09-19 12:33:54.802970: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42162.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_20k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-20000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42162.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42162.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42162.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42164.out:2018-09-19 13:09:44.945851: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42164.out:2018-09-19 13:09:44.946395: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42164.out:2018-09-19 13:09:44.946426: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42164.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_40k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-40000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42164.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42164.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42164.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42145.out:2018-09-19 08:32:57.065848: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42145.out:2018-09-19 08:32:57.066418: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42145.out:2018-09-19 08:32:57.066468: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42145.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_10k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-10000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42145.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42145.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42145.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42155.out:2018-09-19 10:28:21.272231: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42155.out:2018-09-19 10:28:21.272744: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42155.out:2018-09-19 10:28:21.272775: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42155.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_30k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-30000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42155.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42155.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42155.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42150.out:2018-09-19 09:22:34.321065: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42150.out:2018-09-19 09:22:34.321674: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42150.out:2018-09-19 09:22:34.321705: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42150.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_60k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-60000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42150.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42150.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42150.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42154.out:2018-09-19 10:10:19.828221: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42154.out:2018-09-19 10:10:19.828814: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42154.out:2018-09-19 10:10:19.828844: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42154.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_20k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-20000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42154.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42154.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42154.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42148.out:2018-09-19 09:02:44.466001: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42148.out:2018-09-19 09:02:44.466574: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42148.out:2018-09-19 09:02:44.466608: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42148.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_40k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-40000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42148.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42148.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42148.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42157.out:2018-09-19 11:04:20.591425: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42157.out:2018-09-19 11:04:20.591937: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42157.out:2018-09-19 11:04:20.591969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42157.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_50k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-50000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42157.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42157.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42157.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42149.out:2018-09-19 09:12:38.359272: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42149.out:2018-09-19 09:12:38.360065: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42149.out:2018-09-19 09:12:38.360116: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42149.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_50k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-50000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42149.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42149.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42149.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42151.out:2018-09-19 09:32:30.332321: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42151.out:2018-09-19 09:32:30.333086: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42151.out:2018-09-19 09:32:30.333123: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42151.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_70k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-70000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42151.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42151.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42151.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42147.out:2018-09-19 08:52:49.372045: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42147.out:2018-09-19 08:52:49.372626: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42147.out:2018-09-19 08:52:49.372656: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42147.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_30k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-30000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42147.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42147.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42147.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42159.out:2018-09-19 11:40:14.689788: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42159.out:2018-09-19 11:40:14.690275: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42159.out:2018-09-19 11:40:14.690305: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42159.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_70k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-70000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42159.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42159.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42159.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42156.out:2018-09-19 10:46:22.951064: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42156.out:2018-09-19 10:46:22.951647: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42156.out:2018-09-19 10:46:22.951677: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42156.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D3_40k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_600/VGGnet_fast_rcnn-40000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42156.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42156.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42156.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42152.out:2018-09-19 09:42:23.824362: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42152.out:2018-09-19 09:42:23.824980: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42152.out:2018-09-19 09:42:23.825015: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42152.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D1_80k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-80000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42152.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42152.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42152.out:/gpu:0
experiments/scripts/plots/loss/test/slurm-42165.out:2018-09-19 13:27:36.948474: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/test/slurm-42165.out:2018-09-19 13:27:36.948985: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/test/slurm-42165.out:2018-09-19 13:27:36.949015: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/test/slurm-42165.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_D4_50k', learning_rate=0.0, max_iters=9220, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_600/VGGnet_fast_rcnn-50000', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/test/slurm-42165.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/test/slurm-42165.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/test/slurm-42165.out:/gpu:0
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_60k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_70k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_30k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_50k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_40k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_10k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_60k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_30k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_20k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_60k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_30k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_80k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_50k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_10k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_80k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_20k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_20k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_40k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D3_70k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_10k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D1_50k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_70k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_80k.sh:                    --device gpu \
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:#SBATCH --partition=mlgpu
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:#SBATCH --gres=gpu:1
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/plots/loss/slurm_script/D4_40k.sh:                    --device gpu \
experiments/scripts/plots/loss/train/D3_132_train.out:2018-09-14 11:23:47.546862: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D3_132_train.out:2018-09-14 11:23:47.547350: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D3_132_train.out:2018-09-14 11:23:47.547380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D3_132_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD3', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D3_132_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D3_132_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D3_132_train.out:/gpu:0
experiments/scripts/plots/loss/train/D4_600_train.out:2018-09-19 06:01:08.885283: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D4_600_train.out:2018-09-19 06:01:08.885880: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D4_600_train.out:2018-09-19 06:01:08.885911: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D4_600_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD4', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D4_600_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D4_600_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D4_600_train.out:/gpu:0
experiments/scripts/plots/loss/train/D3_600_train.out:2018-09-06 10:27:29.243606: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D3_600_train.out:2018-09-06 10:27:29.244143: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D3_600_train.out:2018-09-06 10:27:29.244182: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D3_600_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD3', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D3_600_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D3_600_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D3_600_train.out:/gpu:0
experiments/scripts/plots/loss/train/D1_264_train.out:2018-09-05 17:50:24.989099: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D1_264_train.out:2018-09-05 17:50:24.989681: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D1_264_train.out:2018-09-05 17:50:24.989711: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D1_264_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD1_264', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D1_264_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D1_264_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D1_264_train.out:/gpu:0
experiments/scripts/plots/loss/train/D3_264_train.out:2018-09-06 05:46:54.937206: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D3_264_train.out:2018-09-06 05:46:54.937727: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D3_264_train.out:2018-09-06 05:46:54.937758: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D3_264_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD3_264', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D3_264_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D3_264_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D3_264_train.out:/gpu:0
experiments/scripts/plots/loss/train/D1_132_train.out:2018-09-14 09:23:29.300764: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D1_132_train.out:2018-09-14 09:23:29.301293: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D1_132_train.out:2018-09-14 09:23:29.301325: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D1_132_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD1', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D1_132_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D1_132_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D1_132_train.out:/gpu:0
experiments/scripts/plots/loss/train/D4_132_train.out:2018-09-14 10:24:49.877090: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D4_132_train.out:2018-09-14 10:24:49.877673: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D4_132_train.out:2018-09-14 10:24:49.877703: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D4_132_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD4', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D4_132_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D4_132_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D4_132_train.out:/gpu:0
experiments/scripts/plots/loss/train/D4_264_train.out:2018-09-05 19:13:22.507928: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D4_264_train.out:2018-09-05 19:13:22.508416: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D4_264_train.out:2018-09-05 19:13:22.508532: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D4_264_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD4_264', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D4_264_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D4_264_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D4_264_train.out:/gpu:0
experiments/scripts/plots/loss/train/D1_600_train.out:2018-09-15 03:15:40.281945: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/loss/train/D1_600_train.out:2018-09-15 03:15:40.282466: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/loss/train/D1_600_train.out:2018-09-15 03:15:40.282498: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/loss/train/D1_600_train.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', device='gpu', device_id=0, imdb_name='rgz_2017_trainD1', learning_rate=0.0, max_iters=80000, network_name='rgz_train', pretrained_model='/home/yuno/intern/rgz_rcnn/data/pretrained_model/imagenet/VGG_imagenet.npy', randomize=False, set_cfgs=None, solver=None, start_iter=0)
experiments/scripts/plots/loss/train/D1_600_train.out: 'GPU_ID': 0,
experiments/scripts/plots/loss/train/D1_600_train.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/loss/train/D1_600_train.out:/gpu:0
experiments/scripts/plots/accuracy/D4_264_test.out:2018-09-06 09:51:25.805862: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D4_264_test.out:2018-09-06 09:51:25.806347: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D4_264_test.out:2018-09-06 09:51:25.806377: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D4_264_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD4', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_264/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D4_264_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D4_264_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D4_264_test.out:/gpu:0
experiments/scripts/plots/accuracy/D3_264_test.out:2018-09-06 09:47:49.626810: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D3_264_test.out:2018-09-06 09:47:49.627297: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D3_264_test.out:2018-09-06 09:47:49.627327: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D3_264_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD3', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_264/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D3_264_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D3_264_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D3_264_test.out:/gpu:0
experiments/scripts/plots/accuracy/D3_600_test.out:2018-09-14 08:27:18.830276: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D3_600_test.out:2018-09-14 08:27:18.830831: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D3_600_test.out:2018-09-14 08:27:18.830862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D3_600_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD3', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D3_600_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D3_600_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D3_600_test.out:/gpu:0
experiments/scripts/plots/accuracy/D4_132_test.out:2018-09-14 12:23:01.781350: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D4_132_test.out:2018-09-14 12:23:01.781993: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D4_132_test.out:2018-09-14 12:23:01.782024: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D4_132_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD4', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4_132/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D4_132_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D4_132_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D4_132_test.out:/gpu:0
experiments/scripts/plots/accuracy/D1_264_test.out:2018-09-06 08:32:42.646096: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D1_264_test.out:2018-09-06 08:32:42.646675: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D1_264_test.out:2018-09-06 08:32:42.646706: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D1_264_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_264.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD1', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_264/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D1_264_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D1_264_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D1_264_test.out:/gpu:0
experiments/scripts/plots/accuracy/D1_600_test.out:2018-09-15 09:16:22.303374: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D1_600_test.out:2018-09-15 09:16:22.303923: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D1_600_test.out:2018-09-15 09:16:22.303958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D1_600_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD1', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_600/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D1_600_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D1_600_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D1_600_test.out:/gpu:0
experiments/scripts/plots/accuracy/D1_132_test.out:2018-09-15 03:11:20.326816: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D1_132_test.out:2018-09-15 03:11:20.327562: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D1_132_test.out:2018-09-15 03:11:20.327609: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D1_132_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD1', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD1_132/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D1_132_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D1_132_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D1_132_test.out:/gpu:0
experiments/scripts/plots/accuracy/D3_132_test.out:2018-09-15 03:12:47.717573: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D3_132_test.out:2018-09-15 03:12:47.718083: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D3_132_test.out:2018-09-15 03:12:47.718112: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D3_132_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end_132.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD3', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD3_132/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D3_132_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D3_132_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D3_132_test.out:/gpu:0
experiments/scripts/plots/accuracy/D4_600_test.out:2018-09-06 10:15:30.844687: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
experiments/scripts/plots/accuracy/D4_600_test.out:2018-09-06 10:15:30.845185: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Found device 0 with properties: 
experiments/scripts/plots/accuracy/D4_600_test.out:2018-09-06 10:15:30.845215: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1120] Creating TensorFlow device (/device:GPU:0) -> (device: 0, name: GeForce GTX 1080 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
experiments/scripts/plots/accuracy/D4_600_test.out:Namespace(cfg_file='/home/yuno/intern/rgz_rcnn/experiments/cfgs/faster_rcnn_end2end.yml', comp_mode=True, device='gpu', device_id=0, force=True, imdb_name='rgz_2017_testD4', model='/home/yuno/intern/rgz_rcnn/output/faster_rcnn_end2end/rgz_2017_trainD4/VGGnet_fast_rcnn-80000', network_name='rgz_test', prototxt=None, thresh=0.05)
experiments/scripts/plots/accuracy/D4_600_test.out: 'GPU_ID': 0,
experiments/scripts/plots/accuracy/D4_600_test.out: 'USE_GPU_NMS': True}
experiments/scripts/plots/accuracy/D4_600_test.out:/gpu:0
experiments/scripts/example_test_slurm.sh:#SBATCH --gres=gpu:1
experiments/scripts/example_test_slurm.sh:module load tensorflow/1.4.0-py27-gpu
experiments/scripts/example_test_slurm.sh:# if cuda driver is not in the system path, customise and add the following paths
experiments/scripts/example_test_slurm.sh:# export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
experiments/scripts/example_test_slurm.sh:# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
experiments/scripts/example_test_slurm.sh:                    --device gpu \
lib/setup.py:def locate_cuda():
lib/setup.py:    """Locate the CUDA environment on the system
lib/setup.py:    Starts by looking for the CUDAHOME env variable. If not found, everything
lib/setup.py:    # first check if the CUDAHOME env variable is in use
lib/setup.py:    if 'CUDAHOME' in os.environ:
lib/setup.py:        home = os.environ['CUDAHOME']
lib/setup.py:        default_path = pjoin(os.sep, 'usr', 'local', 'cuda', 'bin')
lib/setup.py:    cudaconfig = {'home':home, 'nvcc':nvcc,
lib/setup.py:    for k, v in cudaconfig.iteritems():
lib/setup.py:    return cudaconfig
lib/setup.py:CUDA = locate_cuda()
lib/setup.py:if (CUDA):
lib/setup.py:    print('Found cuda lib = {0}'.format(CUDA['lib64']))
lib/setup.py:    print('No CUDA is found, NMS will be compiled using CPUs only')
lib/setup.py:            # use the cuda for .cu files
lib/setup.py:            self.set_executable('compiler_so', CUDA['nvcc'])
lib/setup.py:        # reset the default compiler_so, which we might have changed for cuda
lib/setup.py:if CUDA:
lib/setup.py:        Extension('nms.gpu_nms',
lib/setup.py:            ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
lib/setup.py:            library_dirs=[CUDA['lib64']],
lib/setup.py:            libraries=['cudart'],
lib/setup.py:            runtime_library_dirs=[CUDA['lib64']],
lib/setup.py:            include_dirs = [numpy_include, CUDA['include']]
lib/fast_rcnn/config.py:    # Use GPU implementation of non-maximum suppression
lib/fast_rcnn/config.py:    __C.USE_GPU_NMS = True
lib/fast_rcnn/config.py:    # Default GPU device id
lib/fast_rcnn/config.py:    __C.GPU_ID = 0
lib/fast_rcnn/config.py:    __C.USE_GPU_NMS = False
lib/fast_rcnn/nms_wrapper.py:if cfg.USE_GPU_NMS:
lib/fast_rcnn/nms_wrapper.py:    from nms.gpu_nms import gpu_nms
lib/fast_rcnn/nms_wrapper.py:    """Dispatch to either CPU or GPU NMS implementations."""
lib/fast_rcnn/nms_wrapper.py:    if cfg.USE_GPU_NMS and not force_cpu:
lib/fast_rcnn/nms_wrapper.py:        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
lib/nms/nms_kernel.cu:#include "gpu_nms.hpp"
lib/nms/nms_kernel.cu:#define CUDA_CHECK(condition) \
lib/nms/nms_kernel.cu:  /* Code block avoids redefinition of cudaError_t error */ \
lib/nms/nms_kernel.cu:    cudaError_t error = condition; \
lib/nms/nms_kernel.cu:    if (error != cudaSuccess) { \
lib/nms/nms_kernel.cu:      std::cout << cudaGetErrorString(error) << std::endl; \
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaGetDevice(&current_device));
lib/nms/nms_kernel.cu:  // The call to cudaSetDevice must come before any calls to Get, which
lib/nms/nms_kernel.cu:  // may perform initialization using the GPU.
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaSetDevice(device_id));
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaMalloc(&boxes_dev,
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaMemcpy(boxes_dev,
lib/nms/nms_kernel.cu:                        cudaMemcpyHostToDevice));
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaMalloc(&mask_dev,
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaMemcpy(&mask_host[0],
lib/nms/nms_kernel.cu:                        cudaMemcpyDeviceToHost));
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaFree(boxes_dev));
lib/nms/nms_kernel.cu:  CUDA_CHECK(cudaFree(mask_dev));
lib/nms/gpu_nms.pyx:cdef extern from "gpu_nms.hpp":
lib/nms/gpu_nms.pyx:def gpu_nms(np.ndarray[np.float32_t, ndim=2] dets, np.float thresh,

```
