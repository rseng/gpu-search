# https://github.com/siminegroup/E2EDNA2

```console
simu_config_amoeba.yaml:# Compute platform info: device, OS, procssing platform, CUDA_precision
simu_config_amoeba.yaml:platform        : CPU   # CPU or CUDA or OpenCL
simu_config_amoeba.yaml:CUDA_precision:         # single or double or mixed. only meaningful if CUDA or OpenCL
simu_config_amoeba.yaml:#         if checkpoint file is created on CUDA, does not work on CPU.
e2edna-env.yml:  - cudatoolkit=11.7         # double check the cuda version of the Nvidia driver on your GPU. If no GPU, should comment out this line.
interfaces.py:        if (params['platform'] == 'CUDA') or (params['platform'] == 'OpenCL'):  # 'CUDA' or 'CPU' or 'OpenCL'
interfaces.py:            self.platformProperties = {'Precision': params['CUDA_precision']}
interfaces.py:        if (params['platform'] == 'CUDA') or (params['platform'] == 'OpenCL'):
simu_config.yaml:# Compute platform info: device, OS, procssing platform, CUDA_precision
simu_config.yaml:platform        : CPU   # CPU or CUDA or OpenCL
simu_config.yaml:CUDA_precision:         # single or double or mixed. only meaningful if CUDA or OpenCL
simu_config.yaml:#         if checkpoint file is created on CUDA, does not work on CPU.
examples/automated_tests/simu_config_automated_tests_no_NUPACK_MMB.yaml:# Compute platform info: device, OS, procssing platform, CUDA_precision
examples/automated_tests/simu_config_automated_tests_no_NUPACK_MMB.yaml:platform        : CPU   # CPU or CUDA
examples/automated_tests/simu_config_automated_tests_no_NUPACK_MMB.yaml:CUDA_precision:         # single or double. only meaningful if CUDA
examples/automated_tests/simu_config_automated_tests_no_NUPACK_MMB.yaml:#         if checkpoint file is created on CUDA, does not work on CPU.
examples/automated_tests/simu_config_automated_tests.yaml:# Compute platform info: device, OS, procssing platform, CUDA_precision
examples/automated_tests/simu_config_automated_tests.yaml:platform        : CPU   # CPU or CUDA
examples/automated_tests/simu_config_automated_tests.yaml:CUDA_precision:         # single or double. only meaningful if CUDA
examples/automated_tests/simu_config_automated_tests.yaml:#         if checkpoint file is created on CUDA, does not work on CPU.
simu_config_implicit_solvent.yaml:# Compute platform info: device, OS, procssing platform, CUDA_precision
simu_config_implicit_solvent.yaml:platform        : CPU   # CPU or CUDA or OpenCL
simu_config_implicit_solvent.yaml:CUDA_precision:         # single or double or mixed. only meaningful if CUDA or OpenCL
simu_config_implicit_solvent.yaml:#         if checkpoint file is created on CUDA, does not work on CPU.
README.MD:usage: main.py [-h] -yaml [-ow] [-d] [-os] [-p] [--CUDA_precision] [-w DIR] [-mbdir] [-mb] [--quick_check_mode] [-r] [-m] [-a]
README.MD:  --CUDA_precision    Precision of CUDA, if used (default: single)
README.MD:  * `-p/--platform`: processing platform; either `CPU` or `CUDA`
README.MD:  * `--CUDA_precision`: precision of CUDA if used; either `single` or `double`; Default is `single`
main.py:                         choices=['CPU', 'CUDA', 'OpenCL'])
main.py:compute_info.add_argument('--CUDA_precision', 
main.py:                         help='Precision of CUDA or OpenCL, if used',
main.py:        if params['platform'] in ['CPU', 'CUDA', 'OpenCL']:
main.py:            if params['platform'] == 'CUDA' or params['platform'] == 'OpenCL':
main.py:                if not params['CUDA_precision'] in ['single', 'double', 'mixed']:
main.py:                    parser.error(f"Invalid value: --CUDA_precision={params['CUDA_precision']}. Must be from [single, double, mixed].")
main.py:            parser.error(f"Invalid value: --platform={params['platform']}. Must be from [CPU, CUDA, OpenCL].")

```
