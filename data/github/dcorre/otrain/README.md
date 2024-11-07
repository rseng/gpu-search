# https://github.com/dcorre/otrain

```console
otrain/train.py:from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
otrain/train.py:#from tensorflow.keras.utils import multi_gpu_model
otrain/train.py:    nb_gpus=-1,
otrain/train.py:    # If GPUs are used
otrain/train.py:    if nb_gpus > 0:
otrain/train.py:        parallel_model = multi_gpu_model(model, gpus=nb_gpus)
otrain/train.py:        # save does not work on multi_gpu_model
otrain/train.py:    # If no GPU is used

```
