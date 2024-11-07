# https://github.com/zpliulab/GeoBind

```console
models/geometry_processing.py:# os.environ['CUDA_PATH']='/home/aoli/tools/cuda10.0'
models/geometry_processing.py:                for the KeOps CUDA engine. Defaults to None.
models/GeoBind_model.py:        self.gpu_ids = opt.device
models/GeoBind_model.py:        self.device = torch.device('{}'.format(self.gpu_ids)) if self.gpu_ids else torch.device('cpu')
models/GeoBind_model.py:        torch.cuda.synchronize(device=self.device)
models/GeoBind_model.py:        torch.cuda.reset_max_memory_allocated(device=self.device)
models/GeoBind_model.py:        torch.cuda.synchronize(device=self.device)
models/GeoBind_model.py:        memory_usage = torch.cuda.max_memory_allocated(device=self.device)
models/GeoBind_model.py:        if torch.cuda.is_available():
models/GeoBind_model.py:            self.cuda(self.device)
predict.sh:--device cuda:0 \
Arguments.py:    "--device", type=str, default="cuda:0", help="Which gpu/cpu to train on"
Dataset_lists/Ligands_by_DELIA_ATPbind/MN-440_Train.txt:4gpu:A:A1	A_A E223 D262 E273 I274 K275
README.md:* [Pytorch*](https://pytorch.org/) (v1.10.1). Pytorch with GPU version. Use to model, train, and evaluate the actual neural networks.
train.sh:--device cuda:1 \

```
