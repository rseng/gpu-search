# https://github.com/bruce-chappell/SunnyNet

```console
SunnyNet.py:                         loss_function='MSELoss', alpha=1e-3, cuda=True):
SunnyNet.py:    cuda : bool, optional
SunnyNet.py:        Whether to use GPU acceleration through CUDA (default True).
SunnyNet.py:        'cuda': {'use_cuda': cuda, 'multi_gpu': False},
SunnyNet.py:    print('CUDA VERSION: ', torch.version.cuda)
SunnyNet.py:    print(f'CUDA available: {torch.cuda.is_available()}')
SunnyNet.py:    if torch.cuda.is_available():
SunnyNet.py:        print('GPU name:', torch.cuda.get_device_name())
SunnyNet.py:        print(f'Number of GPUS: {torch.cuda.device_count()}')
SunnyNet.py:                                 cuda=True, model_type='SunnyNet_3x3', loss_function='MSELoss',
SunnyNet.py:    cuda : bool, optional
SunnyNet.py:        Whether to use GPU acceleration through CUDA (default True).
SunnyNet.py:        'cuda': cuda,      
SunnyNet.py:        'multi_gpu_train': False,
networkUtils/lossFunctions.py:    'device': torch.device()                # either 'cuda' or 'cpu'
networkUtils/atmosphereFunctions.py:        'cuda': (bool),                # whether to use CUDA for forward pass 
networkUtils/atmosphereFunctions.py:        'multi_gpu_train': (bool),     # whether the model was traine on multiple GPUs or just 1
networkUtils/atmosphereFunctions.py:    ## fix dict keys from multi gpu training ##
networkUtils/atmosphereFunctions.py:    if config['multi_gpu_train']:
networkUtils/atmosphereFunctions.py:    if config['cuda']: 
networkUtils/atmosphereFunctions.py:        model.network.to('cuda')
networkUtils/modelWrapper.py:        'cuda': {'use_cuda': (bool), 'multi_gpu': (bool)}, # whether to use cuda and multi GPU when training
networkUtils/modelWrapper.py:            ## set CPU/GPU ##
networkUtils/modelWrapper.py:            if params['cuda']['use_cuda']:
networkUtils/modelWrapper.py:                self.device = torch.device("cuda:0")
networkUtils/modelWrapper.py:                if params['cuda']['multi_gpu']:    
networkUtils/modelWrapper.py:                    if torch.cuda.device_count() > 1:
networkUtils/modelWrapper.py:                        print(f" Using {torch.cuda.device_count()} GPUs")
networkUtils/modelWrapper.py:                        print(f"Using 1 GPU")
networkUtils/modelWrapper.py:                    print(f"Using 1 GPU")
networkUtils/modelWrapper.py:            ## send to CPU/GPU ##
networkUtils/modelWrapper.py:            ## set CPU/GPU ##
networkUtils/modelWrapper.py:            if params['cuda']:
networkUtils/modelWrapper.py:                self.device = "cuda"
README.md:                              '3D_sim_train_s123.pt', model_type='SunnyNet_3x3',alpha=0.2, cuda=True)

```
