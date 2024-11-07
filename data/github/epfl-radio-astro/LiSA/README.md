# https://github.com/epfl-radio-astro/LiSA

```console
pipelines/run_SDC2_ldev_pipeline.sbatch:#SBATCH --partition gpu
pipelines/run_SDC2_ldev_pipeline.sbatch:#SBATCH --gres gpu:1
utils/train_classifier_CNN.py:    #with tf.device("device:GPU:0"): #/device:XLA_GPU:0
utils/train_regressor_CNN.py:    #with tf.device("device:GPU:0"): #/device:XLA_GPU:0

```
