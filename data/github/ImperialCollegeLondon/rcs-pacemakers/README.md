# https://github.com/ImperialCollegeLondon/rcs-pacemakers

```console
README.md:- You have access to multi-GPU nodes and several models of GPU ([details](https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/computing/job-sizing-guidance/gpu/))
README.md:1. Create a new server (GPU recommended)
README.md:### Multi-GPU execution
README.md:This repository has two branches. The `master` branch (the default) targets a single GPU. The [`multi-gpu`](https://github.com/ImperialCollegeLondon/rcs-pacemakers/tree/multi-gpu) branch uses [`DataParallel`](https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html) to target two GPUs. You can see the relevant modifications by [comparing the branches](https://github.com/ImperialCollegeLondon/rcs-pacemakers/compare/multi-gpu).
pacemakers.pbs.sh:#PBS -l walltime=01:00:00,select=1:ncpus=4:mem=24G:ngpus=1:gpu_type=P100

```
