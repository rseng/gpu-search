# https://github.com/graphnet-team/graphnet

```console
setup.py:    "Environment :: GPU",
docs/source/getting_started/getting_started.md:        gpus=None,
docs/source/getting_started/getting_started.md:        gpus=[0],
docs/source/models/models.rst:        gpus=None,
docs/source/models/models.rst:        gpus=[0],
docs/source/installation/install.rst:   We recommend installing |graphnet|\ GraphNeT without GPU in clean metaprojects.
tests/utilities/test_argparse.py:            "gpus",
tests/utilities/test_argparse.py:            ["--gpus", "2", "3", "--num-workers", "0"]
tests/utilities/test_argparse.py:        self.assertEquals(tuple(args.gpus), (2, 3))
paper/paper.md:The `GraphNeT` framework provides the end-to-end tools to train and deploy GNN reconstruction models. `GraphNeT` leverages industry-standard tools such as `pytorch` [@NEURIPS2019_9015], `PyG` [@Fey_Fast_Graph_Representation_2019], `lightning` [@Falcon_PyTorch_Lightning_2019], and `wandb` [@wandb] for building and training GNNs as well as particle physics standard tools such as `awkward` [@jim_pivarski_2020_3952674] for handling the variable-size data representing particle interaction events in neutrino telescopes. The inference speed on a single GPU allows for processing the full online datastream of current neutrino telescopes in real-time.
requirements/torch_cu118.txt:# Contains packages requirements for GPU installation
requirements/torch_cu121.txt:# Contains packages requirements for GPU installation
docker/gnn-benchmarking/dockerfile:ARG CUDA=cpu
docker/gnn-benchmarking/dockerfile:RUN pip3 install torch==${TORCH}+${CUDA} -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
docker/gnn-benchmarking/dockerfile:    pip3 install torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
docker/gnn-benchmarking/dockerfile:    pip3 install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html && \
docker/gnn-benchmarking/dockerfile:    pip3 install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html  && \
docker/gnn-benchmarking/dockerfile:    pip3 install torch-spline-conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html  && \
examples/04_training/05_train_RNN_TITO.py:    gpus: Optional[List[int]],
examples/04_training/05_train_RNN_TITO.py:            "gpus": gpus,
examples/04_training/05_train_RNN_TITO.py:        gpus=config["fit"]["gpus"],
examples/04_training/05_train_RNN_TITO.py:        "gpus",
examples/04_training/05_train_RNN_TITO.py:        args.gpus,
examples/04_training/07_train_normalizing_flow.py:    gpus: Optional[List[int]],
examples/04_training/07_train_normalizing_flow.py:            "gpus": gpus,
examples/04_training/07_train_normalizing_flow.py:        gpus=config["fit"]["gpus"],
examples/04_training/07_train_normalizing_flow.py:        "gpus",
examples/04_training/07_train_normalizing_flow.py:        args.gpus,
examples/04_training/04_train_multiclassifier_from_configs.py:    gpus: Optional[List[int]],
examples/04_training/04_train_multiclassifier_from_configs.py:            "gpus": gpus,
examples/04_training/04_train_multiclassifier_from_configs.py:    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
examples/04_training/04_train_multiclassifier_from_configs.py:        gpus=config.fit["gpus"],
examples/04_training/04_train_multiclassifier_from_configs.py:        "gpus",
examples/04_training/04_train_multiclassifier_from_configs.py:        args.gpus,
examples/04_training/README.md:# Train using a single GPU
examples/04_training/README.md:(graphnet) $ python examples/04_training/01_train_dynedge.py --gpus 0
examples/04_training/README.md:# Train using multiple GPUs
examples/04_training/README.md:(graphnet) $ python examples/04_training/01_train_dynedge.py --gpus 0 1
examples/04_training/README.md:(graphnet) $ python examples/04_training/03_train_dynedge_from_config.py --gpus 0 \
examples/04_training/02_train_tito_model.py:    gpus: Optional[List[int]],
examples/04_training/02_train_tito_model.py:            "gpus": gpus,
examples/04_training/02_train_tito_model.py:        gpus=config["fit"]["gpus"],
examples/04_training/02_train_tito_model.py:        "gpus",
examples/04_training/02_train_tito_model.py:        args.gpus,
examples/04_training/01_train_dynedge.py:    gpus: Optional[List[int]],
examples/04_training/01_train_dynedge.py:            "gpus": gpus,
examples/04_training/01_train_dynedge.py:        gpus=config["fit"]["gpus"],
examples/04_training/01_train_dynedge.py:        "gpus",
examples/04_training/01_train_dynedge.py:        args.gpus,
examples/04_training/03_train_dynedge_from_config.py:    gpus: Optional[List[int]],
examples/04_training/03_train_dynedge_from_config.py:            "gpus": gpus,
examples/04_training/03_train_dynedge_from_config.py:    # NB: Only log to W&B on the rank-zero process in case of multi-GPU
examples/04_training/03_train_dynedge_from_config.py:        gpus=config.fit["gpus"],
examples/04_training/03_train_dynedge_from_config.py:        "gpus",
examples/04_training/03_train_dynedge_from_config.py:        args.gpus,
examples/04_training/06_train_icemix_model.py:    gpus: Optional[List[int]],
examples/04_training/06_train_icemix_model.py:            "gpus": gpus,
examples/04_training/06_train_icemix_model.py:        gpus=config["fit"]["gpus"],
examples/04_training/06_train_icemix_model.py:        "gpus",
examples/04_training/06_train_icemix_model.py:        args.gpus,
src/graphnet/models/easy_model.py:        gpus: Optional[Union[List[int], int]] = None,
src/graphnet/models/easy_model.py:        if gpus:
src/graphnet/models/easy_model.py:            accelerator = "gpu"
src/graphnet/models/easy_model.py:            devices = gpus
src/graphnet/models/easy_model.py:        gpus: Optional[Union[List[int], int]] = None,
src/graphnet/models/easy_model.py:            gpus=gpus,
src/graphnet/models/easy_model.py:        gpus: Optional[Union[List[int], int]] = None,
src/graphnet/models/easy_model.py:            gpus=gpus,
src/graphnet/models/easy_model.py:        gpus: Optional[Union[List[int], int]] = None,
src/graphnet/models/easy_model.py:            gpus=gpus,
src/graphnet/utilities/argparse.py:        "gpus": {
src/graphnet/utilities/argparse.py:                "Indices of GPUs to use for training (default: %(default)s)"

```
