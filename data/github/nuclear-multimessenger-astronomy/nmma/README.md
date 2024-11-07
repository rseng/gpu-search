# https://github.com/nuclear-multimessenger-astronomy/nmma

```console
nmma/em/analysis.py:    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nmma/tests/analysis.py:        gpus=0,
nmma/mlmodel/embedding.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nmma/mlmodel/normalizingflows.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nmma/mlmodel/dataprocessing.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
doc/Cluster_Resources.md:7. `--gpus` : number of GPUs to request
tools/analysis_slurm.py:        "--gpus",
tools/analysis_slurm.py:        help="Number of GPUs to request",
tools/analysis_slurm.py:    fid.write(f"#SBATCH --gpus {args.gpus}\n")
tools/analysis_slurm.py:        if args.gpus > 0:
tools/analysis_slurm.py:            fid.write("module add gpu/0.15.4\n")
tools/analysis_slurm.py:            fid.write("module add cuda\n")
slurm.sub:#SBATCH --gpus 0

```
