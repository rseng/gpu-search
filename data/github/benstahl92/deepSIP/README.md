# https://github.com/benstahl92/deepSIP

```console
deepSIP/model.py:             device type being used (GPU if available, else CPU)
deepSIP/model.py:        # prefer GPUs
deepSIP/model.py:        if torch.cuda.is_available():
deepSIP/model.py:            GPU = True
deepSIP/model.py:            self.device = torch.device('cuda')
deepSIP/model.py:            torch.cuda.empty_cache()
deepSIP/model.py:            GPU = False
deepSIP/model.py:            print('No GPU available. Using CPU instead...')
deepSIP/model.py:            if GPU:
deepSIP/model.py:                          'cpu' if not GPU else 'cuda')
deepSIP/training.py:             device type being used (GPU if available, else CPU)
deepSIP/training.py:              network to train (may be wrapped in DataParallel if on GPU)
deepSIP/training.py:        # instantiate network using GPU, if available
deepSIP/training.py:        if torch.cuda.is_available():
deepSIP/training.py:            self.device = torch.device('cuda')
deepSIP/training.py:            print('Running on {} GPUs.'.format(torch.cuda.device_count()))
deepSIP/training.py:            torch.cuda.empty_cache()
deepSIP/training.py:            print('No GPU available. Training on CPU...')
deepSIP/utils.py:    seeds are set for torch (including cuda), numpy, random, and PHYTHONHASH;
deepSIP/utils.py:    torch.cuda.manual_seed_all(seed)
deepSIP/dataset.py:             device to push tensors to, defaults to GPU if available
deepSIP/dataset.py:            self.device = torch.device('cuda' if torch.cuda.is_available() \

```
