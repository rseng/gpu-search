# https://github.com/he2016012996/CABnet

```console
test.py:os.environ["CUDA_VISIBLE_DEVICES"] = "7"
train.py:from keras.utils import multi_gpu_model
train.py:os.environ["CUDA_VISIBLE_DEVICES"] = "4"    
train.py:gpu_num=1
train.py:if gpu_num>1:
train.py:    parallel_model = multi_gpu_model(model, gpus=gpu_num)

```
