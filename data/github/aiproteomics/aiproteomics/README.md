# https://github.com/aiproteomics/aiproteomics

```console
data_exploration_notebooks/analyze_predictions.sh:#SBATCH -p gpu
data_exploration_notebooks/analyze_predictions.sh:#SBATCH --gpus=1
data_exploration_notebooks/analyze_predictions.py:# # sbatch: A full node consists of 72 CPU cores, 491520 MiB of memory and 4 GPUs and can be shared by up to 4 jobs.
data_exploration_notebooks/analyze_predictions.py:# # sbatch: By default shared jobs get 6826 MiB of memory per CPU core, unless explicitly overridden with --mem-per-cpu, --mem-per-gpu or --mem.
data_exploration_notebooks/analyze_predictions.py:# # sbatch: You will be charged for 1 GPUs, based on the number of CPUs, GPUs and the amount memory that you've requested.
data_exploration_notebooks/predict.sh:#SBATCH -p gpu
data_exploration_notebooks/predict.sh:#SBATCH --gpus=1
data_exploration_notebooks/predict.sh:module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
data_exploration_notebooks/trainfrag.sh:#SBATCH -p gpu
data_exploration_notebooks/trainfrag.sh:#SBATCH --gpus=1
data_exploration_notebooks/trainfrag.sh:module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0
data_exploration_notebooks/train.py:print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

```
