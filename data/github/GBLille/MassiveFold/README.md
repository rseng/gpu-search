# https://github.com/GBLille/MassiveFold

```console
environment.yml:  - nvidia
environment.yml:  - jaxlib=*=cuda*
environment.yml:  - cuda-nvcc
README.md:automatically batches of structure predictions on GPU, and gathering the results in one global output directory, with a 
README.md:MassiveFold's design (see schematic below) is optimized for GPU cluster usage. It allows fast computing for massive 
README.md:jobs are computed on a single GPU node and their results are then gathered as a single output with each prediction 
README.md:This automatic splitting is also convenient for massive sampling on a single GPU server to manage jobs priorities.  
README.md:2. **structure prediction**: on GPU, structure predictions follow the massive sampling principle. The total number 
README.md:of predictions is divided into smaller batches and each of them is distributed on a single GPU. These jobs wait for the 
README.md:**On a GPU cluster:**  
README.md:cluster to specify GPU type, time limits or the project on which the hours are used:
README.md:    "jeanzay_gpu": "v100",
README.md:    "jeanzay_gpu_with_memory": "v100-32g",
README.md:#SBATCH -C $jeanzay_gpu_with_memory
README.md:We provide headers for the Jean Zay French CNRS national GPU cluster ([IDRIS](http://www.idris.fr/),) 
README.md:    "jeanzay_gpu_with_memory": "v100-32g",
README.md:#SBATCH --gpus-per-node=1
README.md:##SBATCH --qos=qos_gpu-t4               # Uncomment for job requiring more than 20h (max 16 GPUs)
README.md:#SBATCH -C $jeanzay_gpu_with_memory     # GPU type+memory
README.md:ColabFold. A GPU with at least 16 GB RAM is also recommended, knowing that more memory allows to model larger systems. 
README.md:run in parallel as each batch can be computed on a different GPU, if available.  
README.md:***N.B.***: an interest to use `run_massivefold.sh` on a single server with a single GPU is to be able to run massive 
README.md:in total divided into 45 batches**, these batches can therefore be run in parallel on a GPU cluster infrastructure.
README.md:For instance, for the Jean Zay GPU cluster:
README.md:    "jeanzay_gpu_with_memory": "v100-32g",
README.md:many batches, the “use_gpu_relax” and “models_to_relax” parameters are set to “false” and “none” respectively. Indeed, 
README.md:[IDRIS Open Hackathon](http://www.idris.fr/annonces/idris-gpu-hackathon-2023.html), part of the Open Hackathons program. 
README.md:The authors would like to acknowledge OpenACC-Standard.org for their support.
install.sh:    CONDA_OVERRIDE_CUDA="11.8" conda env create -f environment.yml
install.sh:    CONDA_OVERRIDE_CUDA="11.8" conda env create -f mf_colabfold.yml
massivefold/parallelization/jeanzay_AFmassive_params.json:        "jeanzay_gpu": "v100",
massivefold/parallelization/jeanzay_AFmassive_params.json:        "jeanzay_gpu_with_memory": "v100", 
massivefold/parallelization/jeanzay_ColabFold_params.json:        "jeanzay_gpu": "v100",
massivefold/parallelization/jeanzay_ColabFold_params.json:        "jeanzay_gpu_with_memory": "v100", 
massivefold/parallelization/templates/AFmassive/jobarray_multimer.slurm:use_gpu_relax=false
massivefold/parallelization/templates/AFmassive/jobarray_multimer.slurm:    --use_gpu_relax=$${use_gpu_relax}
massivefold/parallelization/templates/AFmassive/jobarray_multimer.slurm:    --use_gpu_relax=$${use_gpu_relax} \
massivefold/parallelization/templates/AFmassive/alignment_multimer.slurm:use_gpu_relax=false
massivefold/parallelization/templates/AFmassive/alignment_multimer.slurm:    --use_gpu_relax=$${use_gpu_relax} 
massivefold/parallelization/templates/AFmassive/alignment_multimer.slurm:    --use_gpu_relax=$${use_gpu_relax} \
massivefold/parallelization/templates/AFmassive/jobarray_monomer_ptm.slurm:use_gpu_relax=false
massivefold/parallelization/templates/AFmassive/jobarray_monomer_ptm.slurm:    --use_gpu_relax=$${use_gpu_relax}
massivefold/parallelization/templates/AFmassive/jobarray_monomer_ptm.slurm:    --use_gpu_relax=$${use_gpu_relax} \
massivefold/parallelization/templates/AFmassive/alignment_monomer_ptm.slurm:use_gpu_relax=false
massivefold/parallelization/templates/AFmassive/alignment_monomer_ptm.slurm:    --use_gpu_relax=$${use_gpu_relax} 
massivefold/parallelization/templates/AFmassive/alignment_monomer_ptm.slurm:    --use_gpu_relax=$${use_gpu_relax} \
massivefold/parallelization/headers/example_header_jobarray_jeanzay.slurm:#SBATCH --gpus-per-node=1
massivefold/parallelization/headers/example_header_jobarray_jeanzay.slurm:##SBATCH --qos=qos_gpu-t4               # Uncomment for job requiring more than 20h (max 16 GPUs)
massivefold/parallelization/headers/example_header_jobarray_jeanzay.slurm:#SBATCH -C $jeanzay_gpu_with_memory     # Use gpu
massivefold/parallelization/headers/example_header_jobarray_jeanzay.slurm:#for gpu memory
massivefold/parallelization/headers/example_header_alignment_jeanzay.slurm:##SBATCH --qos=qos_gpu-t4              # Uncomment for job requiring more than 20h (max 16 GPUs)
mf_colabfold.yml:  - nvidia
mf_colabfold.yml:  - jaxlib=*=cuda*
mf_colabfold.yml:  - cuda-nvcc

```
