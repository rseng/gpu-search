# https://github.com/pikarpov-LANL/Sapsan

```console
README.md:Sapsan can be run on both CPU and GPU. Please follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/) to install the latest version (torch>=1.7.1 & CUDA>=11.0).
sapsan/lib/estimator/torch_backend.py:    - configuring to run either on cpu or gpu
sapsan/lib/estimator/torch_backend.py:        if 'cuda' in str(self.device):
sapsan/lib/estimator/torch_backend.py:        torch.cuda.empty_cache()        
sapsan/lib/estimator/torch_backend.py:            if not next(self.model.parameters()).is_cuda: self.model.to(self.device)
sapsan/lib/estimator/torch_backend.py:            cuda_id = next(self.model.parameters()).get_device()
sapsan/lib/estimator/torch_backend.py:            data    = torch.as_tensor(inputs).cuda(cuda_id)
sapsan/lib/estimator/torch_backend.py:            self.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
sapsan/lib/estimator/torch_backend.py:        if self.ddp and torch.cuda.is_available(): device_to_print = 'parallel cuda'
sapsan/lib/estimator/pimlturb/pimlturb_diagonal_estimator.py:        if predictions.is_cuda:
sapsan/lib/estimator/pimlturb/pimlturb_diagonal_estimator.py:            self.device = torch.device('cuda:%d'%predictions.get_device()) 
sapsan/lib/estimator/pimlturb/pimlturb_diagonal_estimator.py:        if 'cuda' in str(self.device):
sapsan/lib/estimator/pimlturb1d/pimlturb1d_estimator.py:        if predictions.is_cuda:
sapsan/lib/estimator/pimlturb1d/pimlturb1d_estimator.py:            self.device = torch.device('cuda:%d'%predictions.get_device()) 
sapsan/lib/estimator/pimlturb1d/pimlturb1d_estimator.py:        if 'cuda' in str(self.device):
sapsan/lib/estimator/torch_modules/interp1d.py:        Linear 1D interpolation on the GPU for Pytorch.
sapsan/lib/estimator/torch_modules/interp1d.py:        The code will run on GPU if all the tensors provided are on a cuda
sapsan/lib/estimator/torch_modules/gaussian.py:        if tensor.is_cuda:
sapsan/lib/estimator/torch_modules/gaussian.py:            self.device = torch.device('cuda:%d'%tensor.get_device()) 

```
