# https://github.com/artefactory/choice-learn

```console
docs/paper/paper.md:The package stands on Tensorflow [@Abadi:2015] for model estimation, offering the possibility to use fast quasi-Newton optimization algorithm such as L-BFGS [@Nocedal:2006] as well as various gradient-descent optimizers [@Tieleman:2012; @Kingma:2017] specialized in handling batches of data. GPU usage is also possible, which can prove to be time-saving.
tests/unit_tests/models/test_rumnet_unit.py:    GPURUMnet,
tests/unit_tests/models/test_rumnet_unit.py:def test_gpu_rumnet():
tests/unit_tests/models/test_rumnet_unit.py:    """Tests the GPURUMNet model."""
tests/unit_tests/models/test_rumnet_unit.py:    model = GPURUMnet(
choice_learn/models/rumnet.py:class GPURUMnet(PaperRUMnet):
choice_learn/models/rumnet.py:    """GPU-optimized Re-Implementation of the RUMnet model.
choice_learn/models/__init__.py:if len(tf.config.list_physical_devices("GPU")) > 0:
choice_learn/models/__init__.py:    logging.info("GPU detected, importing GPU version of RUMnet.")
choice_learn/models/__init__.py:    from .rumnet import GPURUMnet as RUMnet
choice_learn/models/__init__.py:    logging.info("No GPU detected, importing CPU version of RUMnet.")
poetry.lock:and-cuda = ["nvidia-cublas-cu11 (==11.11.3.6)", "nvidia-cuda-cupti-cu11 (==11.8.87)", "nvidia-cuda-nvcc-cu11 (==11.8.89)", "nvidia-cuda-runtime-cu11 (==11.8.89)", "nvidia-cudnn-cu11 (==8.7.0.84)", "nvidia-cufft-cu11 (==10.9.0.58)", "nvidia-curand-cu11 (==10.3.0.86)", "nvidia-cusolver-cu11 (==11.4.1.48)", "nvidia-cusparse-cu11 (==11.7.5.86)", "nvidia-nccl-cu11 (==2.16.5)", "tensorrt (==8.5.3.1)"]
poetry.lock:tensorflow-gpu = ["tensorflow-gpu (>=2.16.0,<2.17.0)"]
poetry.lock:tensorflow-rocm = ["tensorflow-rocm (>=2.16.0,<2.17.0)"]

```
