# https://github.com/Planet-AI-GmbH/tfaip

```console
tfaip/trainer/params.py:            "LD_LIBRARY_PATH=:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64"
tfaip/trainer/params.py:            'options nvidia "NVreg_RestrictProfilingToAdminUsers=0" to /etc/modprobe.d/nvidia-kernel-common.conf'
tfaip/trainer/params.py:        metadata=pai_meta(mode="flat", help="Parameters to setup the devices such as GPUs and multi GPU training."),
tfaip/util/testing/setup.py:if "CUDA_VISIBLE_DEVICES" not in os.environ:
tfaip/util/testing/setup.py:    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable gpu usage
tfaip/device/device_config.py:The device config sets up the GPUs to use for training.
tfaip/device/device_config.py:def default_gpus():
tfaip/device/device_config.py:    # if the env var CUDA_VISIBLE_DEVICES is set, use this as default, else an empty list
tfaip/device/device_config.py:    devs = os.environ.get("CUDA_VISIBLE_DEVICES", None)
tfaip/device/device_config.py:    """Configuration of the devices (GPUs).
tfaip/device/device_config.py:    Specify which gpus to use either by setting gpus or CUDA_VISIBLE_DEVICES
tfaip/device/device_config.py:    gpus: Optional[List[int]] = field(default=None, metadata=pai_meta(help="List of the GPUs to use."))
tfaip/device/device_config.py:    gpu_auto_tune: bool = field(default=False, metadata=pai_meta(help="Enable auto tuning of the GPUs"))
tfaip/device/device_config.py:    gpu_memory: Optional[int] = field(
tfaip/device/device_config.py:        metadata=pai_meta(help="Limit the per GPU memory in MB. By default the memory will grow automatically"),
tfaip/device/device_config.py:        metadata=pai_meta(help="Distribution strategy for multi GPU, select 'mirror' or 'central_storage'"),
tfaip/device/device_config.py:        gpus = params.gpus if params.gpus is not None else default_gpus()
tfaip/device/device_config.py:        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1" if self._params.gpu_auto_tune else "0"
tfaip/device/device_config.py:        physical_gpu_devices = tf.config.list_physical_devices("GPU")
tfaip/device/device_config.py:            physical_gpu_devices = [physical_gpu_devices[i] for i in gpus]
tfaip/device/device_config.py:                f"GPU device not available. Number of devices detected: {len(physical_gpu_devices)}"
tfaip/device/device_config.py:        tf.config.experimental.set_visible_devices(physical_gpu_devices, "GPU")
tfaip/device/device_config.py:        for physical_gpu_device in physical_gpu_devices:
tfaip/device/device_config.py:            tf.config.experimental.set_memory_growth(physical_gpu_device, self._params.gpu_memory is None)
tfaip/device/device_config.py:            if self._params.gpu_memory is not None:
tfaip/device/device_config.py:                tf.config.experimental.set_memory_growth(physical_gpu_device, False)
tfaip/device/device_config.py:                    physical_gpu_device, [tf.config.LogicalDeviceConfiguration(memory_limit=self._params.gpu_memory)]
tfaip/device/device_config.py:        physical_gpu_device_names = ["/gpu:" + d.name.split(":")[-1] for d in physical_gpu_devices]
tfaip/device/device_config.py:            self.strategy = tf.distribute.experimental.CentralStorageStrategy(compute_devices=physical_gpu_device_names)
tfaip/device/device_config.py:            self.strategy = tf.distribute.MirroredStrategy(devices=physical_gpu_device_names)
tfaip/data/__init__.py:# which highly slows down the processing time and might lead to conflicts with GPUs
tfaip/scripts/xlsxexperimenter/README.md:Pass an xlsx file to parse `--xlsx`, the desired `--gpus` for scheduling, 
tfaip/scripts/xlsxexperimenter/README.md:* Run on gpus 1 and 3, `tfaip-experimenter --xlsx demo.xlsx --gpus 1 3`
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:    is_gpu: bool
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:    def __init__(self, xlsx_path, gpus=None, cpus=None, dry_run=False, python=None, use_ts=False, update_mode=False):
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:        gpus = gpus if gpus else []
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:            # error if no cpus or gpus are selected
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:            assert not (use_ts and not gpus and not cpus)
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:            # error if cpus and gpus are selected simultaneously
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:            assert not (gpus and cpus)
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:        self.with_gpu = gpus and len(gpus) > 0
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:        self.devices = [Device(int(gpu[0]), gpu, True) for gpu in gpus] + [Device(int(d[0]), d, False) for d in cpus]
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:                elif self.with_gpu:
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:                    ts_socket = "gpu{}".format(self.devices[device_idx].label)
tfaip/scripts/xlsxexperimenter/run_xlsx_experimenter.py:                    + (["--device.gpus", str(self.devices[device_idx].device_id)] if self.with_gpu else [])
tfaip/scripts/experimenter.py:        "--gpus",
tfaip/scripts/experimenter.py:        help="The gpus to use. For multiple runs on the same gpu use e.g. --gpus 3a 3b 3b",
tfaip/scripts/experimenter.py:        if not args.no_use_tsp and not args.gpus and not args.cpus:
tfaip/scripts/experimenter.py:            raise ValueError("No devices (gpu or cpu) found. Disable task spooler (--use_no_tsp) or use --gpus --cpus")
tfaip/scripts/experimenter.py:        if args.gpus and args.cpus:
tfaip/scripts/experimenter.py:            raise ValueError("Do not mix gpu and cpu calls.")
tfaip/scripts/experimenter.py:    exp = XLSXExperimenter(args.xlsx, args.gpus, args.cpus, args.dry_run, args.python, not args.no_use_tsp, args.update)
docs/source/doc.device_config.rst:The structure also allows to modify the ``DistributionStrategy`` when training on several GPUs.
docs/source/doc.device_config.rst:Selecting GPUs
docs/source/doc.device_config.rst:By default, |tfaip| will not use any GPU.
docs/source/doc.device_config.rst:Modify the used GPUs by setting the ``DeviceConfigParams.gpus`` flag which expects a list of GPU indices, e.g.
docs/source/doc.device_config.rst:    --trainer.device.gpus 0 2
docs/source/doc.device_config.rst:to use the GPUs with index 0 and 2.
docs/source/doc.device_config.rst:Alternatively, if the environment variable ``CUDA_VISIBLE_DEVICES`` is set, |tfaip| will use these GPUs for training.
docs/source/doc.device_config.rst:Multi-GPU setup
docs/source/doc.training.rst:In many cases this leads to improved results on the drawback that more GPU-memory is required during training.
docs/source/doc.training.rst:Example: ``--device.gpus 0 1``.
docs/source/doc.parameters.rst:        gpus: List[int] = field(default_factory=list, metadata=pai_meta(help="GPUs to use"))
docs/source/doc.evaluation.rst:    --lav.device.gpus 0
docs/source/doc.installation.rst:* (optional) cuda/cudnn libs for GPU support, see `tensorflow <https://www.tensorflow.org/install/source#tested_build_configurations>`_ for the versions which are required/compatible.
docs/source/doc.debugging.rst:The standard way to increase the throughput of a model is to increase its batch size if the memory of a GPU is not exceeded: ``--train.batch_size 32``.
test/scripts/experimenter_test_files/1/train.log:    "gpus": null,
test/scripts/experimenter_test_files/1/train.log:    "gpu_auto_tune": false,
test/scripts/experimenter_test_files/1/train.log:    "gpu_memory": null,
test/scripts/experimenter_test_files/1/train.log:INFO     2021-09-22 08:53:31,628     tfaip.device.device_config: Setting up device config DeviceConfigParams(gpus=None, gpu_auto_tune=False, gpu_memory=None, soft_device_placement=True, dist_strategy=<DistributionStrategy.DEFAULT: 'default'>)
test/scripts/experimenter_test_files/2/train.log:    "gpus": null,
test/scripts/experimenter_test_files/2/train.log:    "gpu_auto_tune": false,
test/scripts/experimenter_test_files/2/train.log:    "gpu_memory": null,
test/scripts/experimenter_test_files/2/train.log:INFO     2021-09-22 08:53:31,628     tfaip.device.device_config: Setting up device config DeviceConfigParams(gpus=None, gpu_auto_tune=False, gpu_memory=None, soft_device_placement=True, dist_strategy=<DistributionStrategy.DEFAULT: 'default'>)
README.md:# If you have a GPU, select it by specifying its ID
README.md:tfaip-train examples.tutorial.full --device.gpus 0
examples/imageclassification/README.md:tfaip-train examples.imageclassification --trainer.output_dir ic_model --device.gpus 0  # to run training on the first GPU, if available
examples/imageclassification/README.md:tfaip-train examples.imageclassification --model.conv_filters 30 50 60 --model.dense 200 200 --trainer.output_dir ic_model --device.gpus 0  # try a different (larger) model
examples/atr/README.md:tfaip-train examples.atr --trainer.output_dir atr_model --device.gpus 0  # to run training on the first GPU, if available

```
