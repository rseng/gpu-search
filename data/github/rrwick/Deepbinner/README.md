# https://github.com/rrwick/Deepbinner

```console
setup.py:                      'tensorflow-gpu': ['tensorflow-gpu'],
deepbinner/train_network.py:      on the GPU (more efficient).
README.md:Its most complex requirement is [TensorFlow](https://www.tensorflow.org/), which powers the neural network. TensorFlow can run on CPUs (easy to install, supported on many machines) or on NVIDIA GPUs (better performance). If you're only going to use Deepbinner to classify reads, you may not need GPU-level performance ([read more here](#performance)). But if you want to train your own Deepbinner neural network, then using a GPU is a necessity.
README.md:The simplest way to install TensorFlow for your CPU is with `pip3 install tensorflow`. Building TensorFlow from source may give slighly better performance (because it will use all instructions sets supported by your CPU) but [the installation is more complex](https://www.tensorflow.org/install/install_sources). If you are using Ubuntu and have an NVIDIA GPU, [check out these instructions](https://www.tensorflow.org/install/install_linux#tensorflow_gpu_support) for installing TensorFlow with GPU support.
README.md:[Building TensorFlow from source](https://www.tensorflow.org/install/install_sources) may [give better performance](https://www.tensorflow.org/performance/performance_guide#optimizing_for_cpu) (because it can then use all available instruction sets on your CPU). Running TensorFlow on a GPU will definitely give better Deepbinner performance: my tests on a Tesla K80 could classify over 100 reads/sec.
README.md:* A fast computer to train on, ideally with [TensorFlow running on a big GPU](https://www.tensorflow.org/install/install_linux#NVIDIARequirements).

```
