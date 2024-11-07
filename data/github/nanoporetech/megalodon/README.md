# https://github.com/nanoporetech/megalodon

```console
README.rst:Nanopore basecalling is compute intensive and thus it is highly recommended that GPU resources are specified (``--devices``) for optimal Megalodon performance.
README.rst:    #   Compute settings: GPU devices 0 and 1 with 20 CPU cores
README.rst:    By default, Megalodon assumes Guppy (Linux GPU) is installed in the current working directory (i.e. ``./ont-guppy/bin/guppy_basecall_server``).
README.rst:For example to optimize GPU usage, the following option might be specified: ``--guppy-params "--num_callers 5 --ipc_threads 6"``
setup.cfg:    Environment :: GPU
docs/variant_phasing.rst:   gpu_devices="0 1"
docs/variant_phasing.rst:       --processes $nproc --devices $gpu_devices \
docs/index.rst:Nanopore basecalling is compute intensive and thus it is highly recommended that GPU resources are specified (``--devices``) for optimal Megalodon performance.
docs/index.rst:    #   Compute settings: GPU devices 0 and 1 with 40 CPU cores
docs/index.rst:    By default, Megalodon assumes Guppy (Linux GPU) is installed in the current working directory (i.e. ``./ont-guppy/bin/guppy_basecall_server``).
docs/advanced_arguments.rst:      - This requires a Taiyaki installation (potentially with GPU settings).
docs/advanced_arguments.rst:  - Allows a global cap on GPU memory usage.
docs/extras_aggregate.rst:The ``megalodon`` command, running the basecalling backend, generally requires GPU resources, while the aggregation step generally requires fast disk (SSDs) and a lot of CPU cores.
docs/common_arguments.rst:  - GPU devices to use for basecalling acceleration.
docs/common_arguments.rst:  - Device names can be provided in the following formats: ``0``, ``cuda0`` or ``cuda:0``.
test/test_api.py:        help="GPU devices for guppy basecalling backend.",
test/test_megalodon.sh:#   - useful to minimize time on GPU compute resources
megalodon/backends.py:    if re.match("cuda[0-9]+", device) is not None:
megalodon/backends.py:        return "cuda:{}".format(device[4:])
megalodon/backends.py:    elif not device.startswith("cuda"):
megalodon/backends.py:        return "cuda:{}".format(device)
megalodon/backends.py:        - `prep_model_worker`: Load model onto GPU device
megalodon/backends.py:                    self.torch.cuda.set_device(self.device)
megalodon/backends.py:                    LOGGER.error("Invalid CUDA device: {}".format(device))
megalodon/backends.py:                    raise mh.MegaError("Error setting CUDA GPU device.")
megalodon/backends.py:            self.torch.cuda.empty_cache()
megalodon/__main__.py:            "Only process N chunks concurrently per-read (to avoid GPU memory "
megalodon/__main__.py:        help="GPU devices for guppy or taiyaki basecalling backends.",
megalodon/__main__.py:            "can be useful to minimize time on GPU compute resources. Will "
megalodon_extras/_extras_parsers.py:        help="GPU devices for guppy basecalling backend.",
megalodon_extras/_extras_parsers.py:        help="GPU devices for guppy or taiyaki basecalling backends.",

```
