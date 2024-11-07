# https://github.com/alessiospuriomancini/cosmopower

```console
docs/installation.md:    If you do not have a GPU on your machine, you will see a warning message about it which you can safely ignore.
README.md:    If you do not have a GPU on your machine, you will see a warning message about it which you can safely ignore.
cosmopower/training/training_scripts/cosmopower_NN_CMB_training.py:We will start with a few imports, as well as with checking that the notebook is running on a GPU - this is strongly recommended to speed up training.
cosmopower/training/training_scripts/cosmopower_NN_CMB_training.py:# checking that we are using a GPU
cosmopower/training/training_scripts/cosmopower_NN_CMB_training.py:device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
cosmopower/training/training_scripts/cosmopower_PCAplusNN_CMB_training.py:We will start with a few imports, as well as with checking that the notebook is running on a GPU - this is strongly recommended to speed up the training process.
cosmopower/training/training_scripts/cosmopower_PCAplusNN_CMB_training.py:# checking that we are using a GPU
cosmopower/training/training_scripts/cosmopower_PCAplusNN_CMB_training.py:device = 'gpu:0' if tf.test.is_gpu_available() else 'cpu'
cosmopower/training/training_scripts/cosmopower_PCAplusNN_CMB_training.py:with tf.device('/device:GPU:0'): # ensures we are running on a GPU

```
