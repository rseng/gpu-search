# https://github.com/AarhusCosmology/connect_public

```console
docs/workflow.rst:either in a jobscript similar to ``jobscripts/example.js`` or locally with CPUs or GPUs (remember to load ``cuda`` if using GPUs).
source/train_network.py:        ### Define strategy for training on multiple GPUs ###
source/train_network.py:        if len(tf.config.list_physical_devices('GPU')) > 0:
source/train_network.py:                devices=["/gpu:0","/gpu:1"],
README.md:either in a jobscript similar to ```jobscripts/example.js``` or locally with CPUs or GPUs (remember to load ```cuda``` if using GPUs).

```
