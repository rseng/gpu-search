# https://github.com/ZJUFanLab/bulk2space

```console
README.md:The version of pytorch should be suitable to the CUDA version of your machine. You can find the appropriate version on the [PyTorch website](https://pytorch.org/get-started/locally/).
README.md:Here is an example with CUDA11.6:
bulk2space/vae.py:    # net in cuda now
bulk2space/bulk2space.py:                               gpu=0,
bulk2space/bulk2space.py:        used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
bulk2space/bulk2space.py:                              gpu=0,
bulk2space/bulk2space.py:        used_device = torch.device(f"cuda:{gpu}") if gpu >= 0 and torch.cuda.is_available() else torch.device('cpu')
tutorial/handbook.md:    gpu=0,
tutorial/handbook.md:| gpu | The GPU ID. Use cpu if `--gpu < 0` | (int) `0` |

```
