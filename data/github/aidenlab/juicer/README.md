# https://github.com/aidenlab/juicer

```console
AWS/scripts/mega.sh:#load_gpu=""
AWS/scripts/juicer_hiccups.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
AWS/scripts/juicer_hiccups.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
AWS/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
AWS/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
UGER/scripts/juicer_hiccups.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
UGER/scripts/juicer_hiccups.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
UGER/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
UGER/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
PBS/scripts/postprocessing.sh:    #PBS -l nodes=1:ppn=1:gpus=1:GPU 
PBS/scripts/postprocessing.sh:    $load_cuda
PBS/scripts/juicer.sh:load_cuda='module load cuda/7.5.18/gcc/4.4.7'
PBS/scripts/juicer.sh:    export load_cuda="${load_cuda}"
PBS/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
PBS/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
PBS/README:*HiCCUP step has not been tested in this version because of some un-resolved error in loading Jcuda native libraries.
SLURM/scripts/mega.sh:    load_gpu="module load gcccuda/2016a;module load CUDA/8.0.54;" 
SLURM/scripts/mega.sh:    load_gpu="CUDA_VISIBLE_DEVICES=0,1,2,3"
SLURM/scripts/mega.sh:	    sbatch_req="#SBATCH --gres=gpu:kepler:1"
SLURM/scripts/mega.sh:	${load_gpu}
SLURM/scripts/juicer_hiccups.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
SLURM/scripts/juicer_hiccups.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
SLURM/scripts/juicer.sh:	load_gpu="module load gcccuda/2016a;module load CUDA/8.0.44;" 
SLURM/scripts/juicer.sh:	load_gpu="module load gcccuda/2016a;module load CUDA/8.0.54;" 
SLURM/scripts/juicer.sh:    load_gpu="spack load cuda@8.0.61 arch=\`spack arch\` && CUDA_VISIBLE_DEVICES=0,1,2,3"
SLURM/scripts/juicer.sh:	sbatch_req="#SBATCH --gres=gpu:kepler:1"
SLURM/scripts/juicer.sh:	${load_gpu}
SLURM/scripts/juicer.sh:	echo "load: $load_gpu"
SLURM/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
SLURM/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
README.md:### CUDA (for HiCCUPS peak calling)
README.md:You must have an NVIDIA GPU to install CUDA.
README.md:Instructions for installing the latest version of CUDA can be found on the
README.md:[NVIDIA Developer site](https://developer.nvidia.com/cuda-downloads).
README.md:The native libraries included with Juicer are compiled for CUDA 7 or CUDA 7.5.
README.md:Other versions of CUDA can be used, but you will need to download the
README.md:[JCuda](http://www.jcuda.org/downloads/downloads.html).
README.md:For best performance, use a dedicated GPU. You may also be able to obtain
README.md:access to GPU clusters through Amazon Web Services, Google cloud, or a local research
README.md:If you cannot access a GPU, you can run the [CPU version of HiCCUPS](https://github.com/aidenlab/juicer/wiki/CPU-HiCCUPS) directly using the `.hic` file and Juicer Tools.
CPU/common/juicer_hiccups.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
CPU/common/juicer_hiccups.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
CPU/common/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
CPU/common/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
PBS_without_launch/scripts/mega.sh:load_cuda="module load cuda/7.5.18/gcc/4.4.7"
PBS_without_launch/scripts/mega.sh:jid7=$(qsub  -o ${logdir}/hiccups.log -j oe -q batch -N ${groupname}hiccups -l mem=60gb -l walltime=100:00:00 -l nodes=1:ppn=1:gpus=1:GPU -W depend=afterok:${jid6} <<- HICCUPS
PBS_without_launch/scripts/mega.sh:$load_cuda
PBS_without_launch/scripts/mega.sh:echo $PBS_GPUFILE
PBS_without_launch/scripts/mega.sh:jid8=$(qsub -o ${logdir}/arrowhead.log -j oe -q batch -N ${groupname}_ArwHead -l mem=60gb -l walltime=100:00:00 -l nodes=1:ppn=1:gpus=1:GPU -W depend=afterok:${jid6} <<- ARROWHEAD
PBS_without_launch/scripts/mega.sh:$load_cuda
PBS_without_launch/scripts/mega.sh:echo $PBS_GPUFILE
PBS_without_launch/scripts/postprocessing.sh:    #PBS -l nodes=1:ppn=1:gpus=1:GPU 
PBS_without_launch/scripts/postprocessing.sh:    $load_cuda
PBS_without_launch/scripts/juicer.sh:load_cuda='module load cuda/7.5.18/gcc/4.4.7'
PBS_without_launch/scripts/juicer.sh:    export load_cuda="${load_cuda}"
PBS_without_launch/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
PBS_without_launch/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"
PBS_without_launch/README:*HiCCUP step has not been tested in this version because of some un-resolved error in loading Jcuda native libraries.
LSF/scripts/mega.sh:#load_gpu=""
LSF/scripts/juicer.sh:load_cuda="module load dev/cuda/7.0.28"
LSF/scripts/juicer.sh:    bsub -o $topDir/lsf.out -g ${groupname} -q $long_queue -W $long_queue_time -R "rusage[mem=16000]" -R "rusage[ngpus=1]" $waitstring4 -J "${groupname}_postproc" "$load_java; $load_cuda; ${juiceDir}/scripts/juicer_postprocessing.sh -j ${juiceDir}/scripts/juicer_tools -i ${outputdir}/inter_30.hic -m ${juiceDir}/references/motif -g ${genomeID}"
LSF/scripts/juicer_postprocessing.sh:    echo "GPUs are not installed so HiCCUPs cannot be run";
LSF/scripts/juicer_postprocessing.sh:    echo -e "\n(-: Postprocessing successfully completed, maps too sparse to annotate or GPUs unavailable (-:"

```
