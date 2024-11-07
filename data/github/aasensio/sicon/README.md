# https://github.com/aasensio/sicon

```console
train_encdec.py:import nvidia_smi
train_encdec.py:        self.cuda = torch.cuda.is_available()
train_encdec.py:        self.device = torch.device("cuda" if self.cuda else "cpu")
train_encdec.py:        nvidia_smi.nvmlInit()
train_encdec.py:        self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0) 
train_encdec.py:        print("Computing in {0}".format(nvidia_smi.nvmlDeviceGetName(self.handle)))
train_encdec.py:        kwargs = {'num_workers': 4, 'pin_memory': True} if self.cuda else {}
train_encdec.py:            tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle) 
train_encdec.py:            t.set_postfix(loss=loss_L2_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
evaluate_concat.py:        config.gpu_options.allow_growth=True
README.md:    nvidia_smi
README.md:`nvidia_smi` can be installed using the recipe from fastai
README.md:    conda install nvidia-ml-py3 -c fastai
README.md:If you have an NVIDIA GPU on your system, it will make use of it to accelerate
README.md:in a P100 GPU are 90 seconds per epoch. A training with 50 epochs will last
README.md:Within an Anaconda environment, depending if you have a GPU or not the command
README.md:    conda install tensorflow-gpu 
train_concat.py:        config.gpu_options.allow_growth=True
evaluate_encdec.py:        # Check for the availability of a GPU
evaluate_encdec.py:        self.cuda = torch.cuda.is_available()
evaluate_encdec.py:        self.device = torch.device("cuda" if self.cuda else "cpu")
evaluate_encdec.py:        if (self.cuda):

```
