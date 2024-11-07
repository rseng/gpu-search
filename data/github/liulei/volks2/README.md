# https://github.com/liulei/volks2

```console
README.md:- GPU support (PyTorch and CuPy);
README.md:- Full parellelization (with `mpi4py`), optimized for multi nodes GPU clusters.
README.md:- gcc/clang, gfortran, Python3, NumPy, ctypes, matplotlib (required only for plotting), PyTorch or CuPy (for GPU).
README.md:- GPU acceleration. VOLKS2 provides GPU support with Torch and CuPy frameworks. `main()` function in `pfit.py` gives the demo of backend selection. The default is `numpy`, which selects CPU. `torch` and `cupy` select GPU. According to my test, the performance of these two GPU backends are similar. Once GPU are selected. `open_XXX()` will be called for GPU initilization. The calculation procs will be mapped to GPU devices. E.g., proc 1 for dev 0, proc 2 for dev 1, etc. To reduce the kernel launch and data upload overhead of GPU devices,  the data size should be as large as possible. E.g., `t_seg` is set to 4.0 s to 8.0 s for GPU backends. 
README.md:Note: Due to the slightly different implementation of fitting algorithm, if GPU (with PyTorch or CuPy) is used, mbd will be zero. If available, this quantity can be used for solving. However `sp_fit.py` will conduct fine fitting and yield higher accuracy, and is therefore prefered and used as the default quantity for solving. 
pfit.py:    cfg.dev_count   =   torch.cuda.device_count()
pfit.py:    cfg.dev         =   torch.device('cuda:%d' % (cfg.dev_id))
pfit.py:    cfg.dev_count   =   cp.cuda.runtime.getDeviceCount()
pfit.py:    cp.cuda.Device(cfg.dev_id).use()
pfit.py:#    buf_d   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))
pfit.py:        cp.cuda.stream.get_current_stream().synchronize()
pfit.py:        cp.cuda.stream.get_current_stream().synchronize()
pfit.py:        cp.cuda.stream.get_current_stream().synchronize()
pfit.py:#    buf_1   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))
pfit.py:    buf_d   =   torch.from_numpy(buf).cuda(device = cfg.dev).reshape((nap1, cfg.nfreq, cfg.nchan))
pfit.py:        torch.cuda.synchronize()
pfit.py:        torch.cuda.synchronize()
pfit.py:        torch.cuda.synchronize()
pfit.py:#    torch.cuda.empty_cache()
pfit.py:#                np.save('cpu_gpu_compare.npy', (m1, m2))
utils.py:# 0.5 s for CPU, 48 procs, ~4 s for GPU, ~4 procs per card

```
