# https://github.com/optuna/optuna

```console
docs/source/faq.rst:How can I use two GPUs for evaluating two trials simultaneously?
docs/source/faq.rst:If your optimization target supports GPU (CUDA) acceleration and you want to specify which GPU is used in your script,
docs/source/faq.rst:``main.py``, the easiest way is to set ``CUDA_VISIBLE_DEVICES`` environment variable:
docs/source/faq.rst:    # Specify to use the first GPU, and run an optimization.
docs/source/faq.rst:    $ export CUDA_VISIBLE_DEVICES=0
docs/source/faq.rst:    # Specify to use the second GPU, and run another optimization.
docs/source/faq.rst:    $ export CUDA_VISIBLE_DEVICES=1
docs/source/faq.rst:Please refer to `CUDA C Programming Guide <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars>`__ for further details.
tutorial/10_key_features/005_visualization.py:DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tutorial/20_recipes/002_multi_objective.py:DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
CONTRIBUTING.md:Note that the above command might try to install PyTorch without CUDA to your environment even if your environment has CUDA version already.

```
