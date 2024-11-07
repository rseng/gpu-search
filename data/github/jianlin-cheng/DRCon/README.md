# https://github.com/jianlin-cheng/DRCon

```console
features/trRosetta_features_generator/predict_2.py:    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
DRCON_pred.py:model.cuda()
DRCON_pred.py:        features = data['feat'].float().cuda()
DRCON_pred.py:        real_sequence_length = data['sequence_length'].int().cuda()

```
