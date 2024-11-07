# https://github.com/Gaius-Augustus/Tiberius

```console
test_data/test_vit.py:    if args.gpu:
test_data/test_vit.py:        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
test_data/test_vit.py:    parser.add_argument('--gpu', type=str, default='',
test_data/test_vit.py:        help='Number of GPUS to be used.')
docs/README_old.md:On brain cluster, if above does not give you access to GPUs try:
docs/README_old.md:mamba install -c conda-forge cudatoolkit ipykernel cudnn tensorflow==2.10.*
docs/README_old.md:The following example shows how to start a training on the brain cluster using 1 vison node with 4 GPUs:
docs/README_old.md:#SBATCH --gpus=4
docs/install_tensorflow.md:Adapted from https://gretel.ai/blog/install-tensorflow-with-cuda-cdnn-and-gpu-support-in-4-easy-steps
docs/install_tensorflow.md:To install TensorFlow 2.10 with GPU support using Conda, follow these steps:
docs/install_tensorflow.md:    conda install -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
docs/install_tensorflow.md:4. Verify that TensorFlow 2.10 is installed correctly with GPU support:
docs/install_tensorflow.md:    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
README.md:Run Tiberius with the Singularity container (use `-nv` for GPU support):
README.md:Tensorflow should be installed with GPU support. If you are using conda, you can install Tensorflow 2.10 with these [instructions](docs/install_tensorflow.md).
README.md:python3 -m pip install tensorflow[and-cuda]
README.md:If you want to use GPUs, verify that TensorFlow is installed correctly with GPU support:
README.md:python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
README.md:### Running Tiberius on Differnet GPUs
README.md:Tiberius can run on any GPU with at least 8GB of memory. However, you will need to adjust the batch size to match the memory capacity of your GPU using the `--batch_size` argument. Below is a list of recommended batch sizes for different GPUs:
README.md:Here is a list of GPUs to batch siezes:
bin/gene_pred_hmm.py:                        Increases speed and GPU utilization but also memory usage.
bin/predict_gtf.py:# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
bin/write_tfrecord_species.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
bin/unit_tests.py:# don't use GPU for tests
bin/unit_tests.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
bin/train.py:gpus = tf.config.list_physical_devices('GPU')
bin/train.py:#for gpu in gpus:
bin/train.py:    #tf.config.experimental.set_memory_growth(gpu, True)
bin/train.py:"""cluster_res = tf.distribute.cluster_resolver.SlurmClusterResolver(gpus_per_node=4,
bin/train.py:    gpus_per_task=1)"""

```
