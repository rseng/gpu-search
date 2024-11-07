# https://github.com/abs-tudelft/variant-calling-at-scale

```console
README.md:Variant Calling at Scale is a scalable, parallel and efficient implementation of next generation sequencing data pre-processing and variant calling workflows. Our design tightly integrates most pre-processing workflow stages, using Spark built-in functions to sort reads by coordinates, and mark duplicates efficiently. A cluster scaled DeepVariant for both CPU-only and CPU+GPU clusters is also integrated in this workflow. 
scripts/SVCall.sh:#SBATCH -p gpu 
scripts/ADAM.sh:#SBATCH -p gpu
singularity/bionic.def:    apt-get install ocl-icd-* opencl-headers -y
long-reads/download_singularity_images.sh:singularity pull docker://google/deepvariant:1.3.0-gpu

```
