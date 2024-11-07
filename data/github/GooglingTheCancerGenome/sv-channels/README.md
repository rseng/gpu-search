# https://github.com/GooglingTheCancerGenome/sv-channels

```console
svchannels/cross-validations/workflow_4/profile/config.yaml:    --gpus-per-node={resources.gpus}
svchannels/cross-validations/workflow_4/profile/config.yaml:  - partition=gpu
svchannels/cross-validations/workflow_4/profile/config.yaml:  - gpus=1
svchannels/cross-validations/workflow_3/profile/config.yaml:    --gpus-per-node={resources.gpus}
svchannels/cross-validations/workflow_3/profile/config.yaml:  - partition=gpu
svchannels/cross-validations/workflow_3/profile/config.yaml:  - gpus=1
svchannels/cross-validations/workflow_0/profile/config.yaml:    --gpus-per-node={resources.gpus}
svchannels/cross-validations/workflow_0/profile/config.yaml:  - partition=gpu
svchannels/cross-validations/workflow_0/profile/config.yaml:  - gpus=1
svchannels/cross-validations/workflow_1/profile/config.yaml:    --gpus-per-node={resources.gpus}
svchannels/cross-validations/workflow_1/profile/config.yaml:  - partition=gpu
svchannels/cross-validations/workflow_1/profile/config.yaml:  - gpus=1
svchannels/cross-validations/workflow_2/profile/config.yaml:    --gpus-per-node={resources.gpus}
svchannels/cross-validations/workflow_2/profile/config.yaml:  - partition=gpu
svchannels/cross-validations/workflow_2/profile/config.yaml:  - gpus=1
svchannels/utils/R/plot_evaluation_sim.R:require(ggpubr)
svchannels/sandbox/slurm/run_svchannels_loco.sh:	sbatch -J ${CHR}.loco.eval.svchan -p gpu --gpus-per-node=RTX6000:1  -e ${CHR}.loco.eval.err -o ${CHR}.loco.eval.out --export=CHR=${CHR} run_evaluation_loco.slurm
workflow/config.yaml:    --gpus-per-node={resources.gpus_per_node}
workflow/config.yaml:  - gpus_per_node=0
workflow/Snakefile:        partition = "gpu",
workflow/Snakefile:        gpus_per_node = "RTX6000:1",
workflow/Snakefile:        partition = "gpu",
workflow/Snakefile:        gpus_per_node = "RTX6000:1",
workflow_locso/config.yaml:    --gpus-per-node={resources.gpus_per_node}
workflow_locso/config.yaml:  - gpus_per_node=0
workflow_locso/Snakefile:        partition = "gpu",
workflow_locso/Snakefile:        gpus_per_node = "RTX6000:1",
workflow_locso/Snakefile:        partition = "gpu",
workflow_locso/Snakefile:        gpus_per_node = "RTX6000:1",

```
