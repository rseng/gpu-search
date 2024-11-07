# https://github.com/theGreatHerrLebert/rustims

```console
imspy/imspy/simulation/timsim/simulator.py:# don't use all the memory for the GPU (if available)
imspy/imspy/simulation/timsim/simulator.py:gpus = tf.config.experimental.list_physical_devices('GPU')
imspy/imspy/simulation/timsim/simulator.py:if gpus:
imspy/imspy/simulation/timsim/simulator.py:        for i, _ in enumerate(gpus):
imspy/imspy/simulation/timsim/simulator.py:            # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
imspy/imspy/simulation/timsim/simulator.py:                gpus[i],
imspy/imspy/simulation/timsim/simulator.py:            print(f"GPU: {i} memory restricted to 4GB.")
imspy/imspy/simulation/timsim/simulator.py:        # Virtual devices must be set before GPUs have been initialized
imspy/imspy/timstof/dbsearch/imspy_dda.py:# don't use all the memory for the GPU (if available)
imspy/imspy/timstof/dbsearch/imspy_dda.py:gpus = tf.config.experimental.list_physical_devices('GPU')
imspy/imspy/timstof/dbsearch/imspy_dda.py:if gpus:
imspy/imspy/timstof/dbsearch/imspy_dda.py:        for i, _ in enumerate(gpus):
imspy/imspy/timstof/dbsearch/imspy_dda.py:                gpus[i],
imspy/imspy/timstof/dbsearch/imspy_dda.py:            print(f"GPU: {i} memory restricted to 4GB.")
README.md:This will install tensorflow as a dependency without GPU support.
README.md:The easiest way to get GPU support is to additionally install the tensorflow[and-cuda] package:
README.md:pip install tensorflow[and-cuda]==2.15.*
README.md:Which comes with the necessary CUDA and cuDNN libraries.

```
