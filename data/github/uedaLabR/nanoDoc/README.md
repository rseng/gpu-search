# https://github.com/uedaLabR/nanoDoc

```console
README.md:GPU enviroment is strongly recommended to run this software.
README.md:Python (>= 3.6), packages,tensorflow, checked with tensorflow 2.3, cuda/10.1, cudnn/7.6
src/initialtrainingCNN.py:# from tensorflow.keras.utils.training_utils import multi_gpu_model
src/initialtrainingCNN.py:    gpu_count = 4
src/initialtrainingCNN.py:    # model = multi_gpu_model(model, gpus=gpu_count)  # add
src/requirements.txt:faiss-gpu==1.5.3

```
