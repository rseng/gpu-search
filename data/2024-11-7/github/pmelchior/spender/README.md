# https://github.com/pmelchior/spender

```console
setup.py:    install_requires=["torch", "numpy", "accelerate", "torchinterp1d", "astropy", "humanize", "psutil", "GPUtil", "nflows"]
README.md:# if your machine does not have GPUs, specify the device
spender/util.py:import GPUtil
spender/util.py:        if torch.cuda.is_available():
spender/util.py:    if torch.cuda.device_count() == 0:
spender/util.py:    GPUs = GPUtil.getGPUs()
spender/util.py:    for i, gpu in enumerate(GPUs):
spender/util.py:            "GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%".format(
spender/util.py:                i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil * 100
train/train_DESI.py:        print("torch.cuda.device_count():",torch.cuda.device_count())
train/train_normalizing_flow.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train/train_DESI_flow.py:        if torch.cuda.is_available():
train/train_DESI_flow.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train/train_DESI_flow.py:print("torch.cuda.device_count():",torch.cuda.device_count())
train/fp16_train.py:        print("torch.cuda.device_count():",torch.cuda.device_count())

```
