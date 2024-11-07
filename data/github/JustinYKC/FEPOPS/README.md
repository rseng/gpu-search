# https://github.com/JustinYKC/FEPOPS

```console
environment.yml:      - nvidia-cublas-cu11==11.10.3.66
environment.yml:      - nvidia-cuda-cupti-cu11==11.7.101
environment.yml:      - nvidia-cuda-nvrtc-cu11==11.7.99
environment.yml:      - nvidia-cuda-runtime-cu11==11.7.99
environment.yml:      - nvidia-cudnn-cu11==8.5.0.96
environment.yml:      - nvidia-cufft-cu11==10.9.0.58
environment.yml:      - nvidia-curand-cu11==10.2.10.91
environment.yml:      - nvidia-cusolver-cu11==11.4.0.1
environment.yml:      - nvidia-cusparse-cu11==11.7.4.91
environment.yml:      - nvidia-nccl-cu11==2.14.3
environment.yml:      - nvidia-nvtx-cu11==11.7.91
src/fepops/fepops_persistent/fepops_persistent_abc.py:        fepops objects, can be one of "sklearn", "pytorchgpu",
src/fepops/fepops.py:        calculations. May be one of "sklearn", "pytorchgpu", or "pytorchcpu". If
src/fepops/fepops.py:        GPU accelerated mode. Note: GPU accelerated mode should only be used if
src/fepops/fepops.py:        molecules.  Small molecules will not benefit at all from GPU
src/fepops/fepops.py:        kmeans_method: Literal['sklearn', 'pytorchcpu', 'pytorchgpu'] = 'sklearn',
src/fepops/fepops.py:            calculations. May be one of "sklearn", "pytorchgpu", or "pytorchcpu". If
src/fepops/fepops.py:            GPU accelerated mode. Note: GPU accelerated mode should only be used if
src/fepops/fepops.py:            molecules.  Small molecules will not benefit at all from GPU
src/fepops/fepops.py:    def _perform_kmeans_pytorchgpu(
src/fepops/fepops.py:        """Perform kmeans calculation using pytorch (gpu accelerated)
src/fepops/fepops.py:        mol_coors_torch = torch.from_numpy(atom_coords).to("cuda")

```
