# https://github.com/PabloVD/HaloGraphNet

```console
hyperparams_optimization.py:    if torch.cuda.is_available():
hyperparams_optimization.py:        torch.cuda.empty_cache()
Source/training.py:# use GPUs if available
Source/training.py:if torch.cuda.is_available():
Source/training.py:    print("CUDA Available")
Source/training.py:    device = torch.device('cuda')
Source/training.py:    print('CUDA Not Available')

```
