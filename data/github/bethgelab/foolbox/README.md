# https://github.com/bethgelab/foolbox

```console
README.rst:Foolbox is tested with Python 3.8 and newer - however, it will most likely also work with version 3.6 - 3.8. To use it with `PyTorch <https://pytorch.org>`_, `TensorFlow <https://www.tensorflow.org>`_, or `JAX <https://github.com/google/jax>`_, the respective framework needs to be installed separately. These frameworks are not declared as dependencies because not everyone wants to use and thus install all of them and because some of these packages have different builds for different architectures and CUDA versions. Besides that, all essential dependencies are automatically installed.
guide/guide/getting-started.md:Foolbox requires Python 3.8 or newer. To use it with [PyTorch](https://pytorch.org), [TensorFlow](https://www.tensorflow.org), or [JAX](https://github.com/google/jax), the respective framework needs to be installed separately. These frameworks are not declared as dependencies because not everyone wants to use and thus install all of them and because some of these packages have different builds for different architectures and CUDA versions. Besides that, all essential dependencies are automatically installed.
tests/conftest.py:    if not tf.test.is_gpu_available():
tests/conftest.py:        pytest.skip("ResNet50 test too slow without GPU")
foolbox/models/tensorflow.py:        device = tf.device("/GPU:0" if tf.test.is_gpu_available() else "/CPU:0")
foolbox/models/pytorch.py:        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
foolbox/attacks/fast_minimum_norm.py:        x: Batch of arbitrary-size tensors to project, possibly on GPU
foolbox/attacks/sparse_l1_descent_attack.py:        # (otherwise, this is not guaranteed on GPUs, see e.g. PyTorch)
performance/README.md:All experiments were done on an Nvidia GeForce GTX 1080 using the PGD attack.
performance/README.md:Note that Foolbox 3 is faster because **1)** it avoids memory copies between GPU

```
