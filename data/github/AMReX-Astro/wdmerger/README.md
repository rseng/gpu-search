# https://github.com/AMReX-Astro/wdmerger

```console
science/collisions/compile/GNUmakefile:USE_CUDA=TRUE
science/collisions/run.sh:# Abort if we run out of GPU memory.
science/collisions/run.sh:amrex__abort_on_out_of_gpu_memory="1"
science/collisions/run.sh:# Limit GPU memory footprint.
science/wdmerger_II/compile/GNUmakefile:USE_CUDA=TRUE
science/wdmerger_II/run.sh:# Limit GPU memory footprint.
science/ignition/compile/GNUmakefile:USE_CUDA := TRUE
job_scripts/machines.sh:        constraint="gpu"
job_scripts/machines.sh:        gpus_per_task="1"
job_scripts/machines.sh:        gpu_bind="map_gpu:0,1,2,3"
job_scripts/run_utils.sh:        # Number of GPUs per MPI rank.
job_scripts/run_utils.sh:        if [ ! -z $gpus_per_task ]; then
job_scripts/run_utils.sh:            echo "#SBATCH --gpus-per-task $gpus_per_task" >> $dir/$job_script
job_scripts/run_utils.sh:        # Task binding to GPUs within a node.
job_scripts/run_utils.sh:        if [ ! -z $gpu_bind ]; then
job_scripts/run_utils.sh:            echo "#SBATCH --gpu-bind $gpu_bind" >> $dir/$job_script
papers/refs.bib:    title = "{FARGO3D: A New GPU-oriented MHD Code}",
papers/refs.bib:abstract={We describe the AMReX suite of astrophysics codes and their application to modeling problems in stellar astrophysics. Maestro is tuned to efficiently model subsonic convective flows while Castro models the highly compressible flows associated with stellar explosions. Both are built on the block-structured adaptive mesh refinement library AMReX. Together, these codes enable a thorough investigation of stellar phenomena, including Type Ia supernovae and X-ray bursts. We describe these science applications and the approach we are taking to make these codes performant on current and future many-core and GPU-based architectures.}}
papers/wdmerger_I/submit2/paper.tex:strategy sub-optimal for use on many-core processors and GPUs. We have
papers/wdmerger_I/submit2/paper.tex:evaluating the hydrodynamics and microphysics modules on GPUs,
papers/wdmerger_I/submit2/paper.tex:GPUs on certain systems.
papers/wdmerger_I/submit/paper.tex:strategy sub-optimal for use on many-core processors and GPUs. We have
papers/wdmerger_I/submit/paper.tex:evaluating the hydrodynamics and microphysics modules on GPUs,
papers/wdmerger_I/submit/paper.tex:GPUs on certain systems.
papers/wdmerger_I/paper.tex:strategy sub-optimal for use on many-core processors and GPUs. We have
papers/wdmerger_I/paper.tex:evaluating the hydrodynamics and microphysics modules on GPUs,
papers/wdmerger_I/paper.tex:GPUs on certain systems.
papers/ignition/submit2/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA
papers/ignition/submit/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA
papers/ignition/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA
papers/ignition/submit3/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA
papers/ignition/arxiv2/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA
papers/ignition/arxiv/paper.tex:  NVIDIA Corporation, 2788 San Tomas Expressway, Santa Clara, CA, 95051, USA

```
