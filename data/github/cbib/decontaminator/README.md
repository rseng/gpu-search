# https://github.com/cbib/decontaminator

```console
envs/environment_gpu.yml:    - tensorflow-gpu==2.3.2
decontaminator/prepare_ds.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
decontaminator/predict.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
decontaminator/predict2.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""
decontaminator/train.py:os.environ["CUDA_VISIBLE_DEVICES"] = "6"
decontaminator/train.py:print(tf.config.list_physical_devices('GPU'))
README.md:### Training Decontaminator on GPU
README.md:If you plan to train Decontaminator on GPU, please use `environment_gpu.yml` or `requirements_gpu.txt` for dependencies installation.
README.md:Those recipes were tested only on the Linux cluster with multiple GPUs.
README.md:If you plan to train Decontaminator on cluster with multiple GPUs, you will need to uncomment line with
README.md:`CUDA_VISIBLE_DEVICES` variable and replace `""` with `"N"` in header of `train.py`, where N is the number of GPU you want to use:
README.md:os.environ["CUDA_VISIBLE_DEVICES"] = "N"
decontaminator_galaxy/predict.py:os.environ["CUDA_VISIBLE_DEVICES"] = ""

```
