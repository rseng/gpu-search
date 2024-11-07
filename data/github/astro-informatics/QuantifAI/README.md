# https://github.com/astro-informatics/QuantifAI

```console
environment.yml:  - nvidia
environment.yml:  - pytorch-cuda=11.7
environment.yml:  - cudatoolkit-dev
README.md:> This Python package is built on top of PyTorch, so a GPU can considerably accelerate all computations. 
quantifai/utils.py:def to_tensor(z, device="cuda", dtype=torch.float):
quantifai/utils.py:        device (str): The device to place the resulting tensor on (default is 'cuda').
quantifai/utils.py:        device (str): The device to place the resulting operators on (e.g. 'cuda', 'cpu').
paper/Liaudat2023/hypothesis_test_potentials_and_results/ungridded_log.out:NVIDIA A100-PCIE-40GB
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:# Use mac M1/M2 chip GPU
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:    if torch.cuda.is_available():
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:        print(torch.cuda.is_available())
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:        print(torch.cuda.device_count())
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:        print(torch.cuda.current_device())
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:        print(torch.cuda.get_device_name(torch.cuda.current_device()))
paper/Liaudat2023/scripts/UQ_SKROCK_CRR.py:        CRR_dir_name + exp_name, "cuda:0", device_type="gpu"
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:    if torch.cuda.is_available():
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:        print(torch.cuda.is_available())
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:        print(torch.cuda.device_count())
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:        print(torch.cuda.current_device())
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:        print(torch.cuda.get_device_name(torch.cuda.current_device()))
paper/Liaudat2023/scripts/ungridded_vis_experiment.py:            CRR_dir_name + exp_name, "cuda:0", device_type="gpu"
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:# Use mac M1/M2 chip GPU
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:    if torch.cuda.is_available():
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:        print(torch.cuda.is_available())
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:        print(torch.cuda.device_count())
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:        print(torch.cuda.current_device())
paper/Liaudat2023/scripts/UQ_SKROCK_wavelets.py:        print(torch.cuda.get_device_name(torch.cuda.current_device()))

```
