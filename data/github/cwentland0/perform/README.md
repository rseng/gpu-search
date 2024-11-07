# https://github.com/cwentland0/perform

```console
tests/integration_tests/test_rom/test_ml_library/test_tfkeras_library.py:        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
tests/integration_tests/test_rom/test_ml_library/test_tfkeras_library.py:        # TODO: add run_gpu=True test when you figure out how to run tests on GPU
tests/integration_tests/constants.py:        f.write("run_gpu = False \n")
doc/userguide/paramindex.rst:* ``run_gpu``: Boolean flag to determine whether to run decoder/encoder inference on the GPU. Please note that running on the CPU is often faster than running on the GPU for these small 1D problems, as memory movement between the host and device can be extremely slow and all memory movement operations are blocking.
doc/roms/rominputs.rst:   * - ``run_gpu``
doc/roms/rominputs.rst:**NOTE**: if running with ``run_gpu = False`` (making model inferences on the CPU), note that TensorFlow convolutional layers cannot handle a ``channels_first`` format. If your network format conforms to ``io_format = "nchw"``, the code will terminate with an error. This issue could theoretically be fixed by the user by including a permute layer to change the layer input ordering to ``channels_last`` before any convolutional layers, but we err on the side of caution here.
perform/rom/ml_library/tfkeras_library.py:    def init_device(self, run_gpu):
perform/rom/ml_library/tfkeras_library.py:        """Initializes GPU execution, if requested.
perform/rom/ml_library/tfkeras_library.py:        TensorFlow >=2.0 can execute from CPU or GPU, this function does some prep work.
perform/rom/ml_library/tfkeras_library.py:        Passing run_gpu=True will limit GPU memory growth, as unlimited TensorFlow memory allocation can be
perform/rom/ml_library/tfkeras_library.py:        Passing run_gpu=False will guarantee that TensorFlow runs on the CPU. Even if GPUs are available,
perform/rom/ml_library/tfkeras_library.py:            run_gpu: Boolean flag indicating whether to execute TensorFlow functions on an available GPU.
perform/rom/ml_library/tfkeras_library.py:        if run_gpu:
perform/rom/ml_library/tfkeras_library.py:            gpus = tf.config.experimental.list_physical_devices("GPU")
perform/rom/ml_library/tfkeras_library.py:            if gpus:
perform/rom/ml_library/tfkeras_library.py:                    # Currently, memory growth needs to be the same across GPUs
perform/rom/ml_library/tfkeras_library.py:                    for gpu in gpus:
perform/rom/ml_library/tfkeras_library.py:                        tf.config.experimental.set_memory_growth(gpu, True)
perform/rom/ml_library/tfkeras_library.py:                    # Memory growth must be set before GPUs have been initialized
perform/rom/ml_library/tfkeras_library.py:            # If GPU is available, TF will automatically run there
perform/rom/ml_library/tfkeras_library.py:            # 	This forces to run on CPU even if GPU is available
perform/rom/ml_library/tfkeras_library.py:            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
perform/rom/ml_library/tfkeras_library.py:            assert self.run_gpu, "Tensorflow cannot handle channels_first on CPUs"
perform/rom/ml_library/tfkeras_library.py:            pass  # works on GPU or CPU
perform/rom/ml_library/ml_library.py:        run_gpu = catch_input(rom_domain.rom_dict, "run_gpu", False)
perform/rom/ml_library/ml_library.py:        self.init_device(run_gpu)

```
