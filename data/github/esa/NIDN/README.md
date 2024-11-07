# https://github.com/esa/NIDN

```console
docs/source/tutorial.rst:- use_gpu (bool) : true or false. Whether to use gpu for calculations (true) or cpu (false)
nidn/training/run_training.py:    # Enable GPU if desired
nidn/training/run_training.py:    if run_cfg.use_gpu:
nidn/training/run_training.py:        torch.set_default_tensor_type(torch.cuda.FloatTensor)
nidn/training/run_training.py:        torch.cuda.empty_cache()
nidn/materials/material_collection.py:        self.epsilon_matrix = torch.stack(eps_list).squeeze()  # .cuda()
nidn/fdtd/backend.py:The ``cuda`` backends are only available for computers with a GPU.
nidn/fdtd/backend.py:    dict(backends="torch.cuda.float32"),
nidn/fdtd/backend.py:    dict(backends="torch.cuda.float64"),
nidn/fdtd/backend.py:    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
nidn/fdtd/backend.py:    TORCH_CUDA_AVAILABLE = False
nidn/fdtd/backend.py:    # Torch Cuda Backend
nidn/fdtd/backend.py:    if TORCH_CUDA_AVAILABLE:
nidn/fdtd/backend.py:        class TorchCudaBackend(TorchBackend):
nidn/fdtd/backend.py:            """Torch Cuda Backend"""
nidn/fdtd/backend.py:                return torch.ones(shape, device="cuda")
nidn/fdtd/backend.py:                return torch.zeros(shape, device="cuda")
nidn/fdtd/backend.py:                    return arr.clone().to(device="cuda", dtype=dtype)
nidn/fdtd/backend.py:                return torch.tensor(arr, device="cuda", dtype=dtype)
nidn/fdtd/backend.py:                    start, stop + 0.5 * float(endpoint) * delta, delta, device="cuda"
nidn/fdtd/backend.py:            - ``torch.cuda`` (defaults to float64 tensors)
nidn/fdtd/backend.py:            - ``torch.cuda.float16``
nidn/fdtd/backend.py:            - ``torch.cuda.float32``
nidn/fdtd/backend.py:            - ``torch.cuda.float64``
nidn/fdtd/backend.py:    if name.startswith("torch.cuda") and not TORCH_CUDA_AVAILABLE:
nidn/fdtd/backend.py:            "Torch cuda backend is not available.\n"
nidn/fdtd/backend.py:            "Do you have a GPU on your computer?\n"
nidn/fdtd/backend.py:            "Is PyTorch with cuda support installed?"
nidn/fdtd/backend.py:        if dtype == "cuda":
nidn/fdtd/backend.py:            device, dtype = "cuda", "float64"
nidn/fdtd/backend.py:        elif device == "cuda":
nidn/fdtd/backend.py:            backend.__class__ = TorchCudaBackend
nidn/fdtd/backend.py:                "Unknown device '{device}'. Available devices: 'cpu', 'cuda'"
nidn/utils/validate_config.py:                     "target_reflectance_spectrum","target_transmittance_spectrum","freq_distribution","use_gpu","avoid_zero_eps"]
nidn/utils/validate_config.py:    boolean_keys = ["use_regularization_loss","add_noise","use_gpu","avoid_zero_eps",]
nidn/utils/resources/default_config.toml:use_gpu = false
nidn/__init__.py:# Set precision (and potentially GPU)
environment.yml:- cudatoolkit>=11.0.221
README.md:While NIDN does support GPU utilization there are only modest performance benefits to it at time of writing.
README.md:   3. With the default configuration PyTorch with CUDA
README.md:      If this should not happen, comment out `cudatoolkit` in the `environment.yml`.

```
