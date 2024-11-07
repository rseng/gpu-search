# https://github.com/SatelliteShorelines/CoastSeg

```console
3_zoo_workflow.py:            "use_GPU": "0",  # 0 or 1 0 means no GPU
debug_scripts/test_new_models.py:            "use_GPU": "0",
debug_scripts/test_new_models.py:            use_GPU="0",
debug_scripts/test_models.py:            "use_GPU": "0",
debug_scripts/test_models.py:            use_GPU="0",
src/coastseg/zoo_model.py:def get_GPU(num_GPU: str) -> None:
src/coastseg/zoo_model.py:    num_GPU = str(num_GPU)
src/coastseg/zoo_model.py:    if num_GPU == "0":
src/coastseg/zoo_model.py:        logger.info("Not using GPU")
src/coastseg/zoo_model.py:        print("Not using GPU")
src/coastseg/zoo_model.py:        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
src/coastseg/zoo_model.py:    elif num_GPU == "1":
src/coastseg/zoo_model.py:        print("Using single GPU")
src/coastseg/zoo_model.py:        logger.info(f"Using 1 GPU")
src/coastseg/zoo_model.py:        # use first available GPU
src/coastseg/zoo_model.py:        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
src/coastseg/zoo_model.py:    if int(num_GPU) == 1:
src/coastseg/zoo_model.py:        # read physical GPUs from machine
src/coastseg/zoo_model.py:        physical_devices = tf.config.experimental.list_physical_devices("GPU")
src/coastseg/zoo_model.py:        print(f"physical_devices (GPUs):{physical_devices}")
src/coastseg/zoo_model.py:        logger.info(f"physical_devices (GPUs):{physical_devices}")
src/coastseg/zoo_model.py:            # Restrict TensorFlow to only use the first GPU
src/coastseg/zoo_model.py:                tf.config.experimental.set_visible_devices(physical_devices, "GPU")
src/coastseg/zoo_model.py:        # disable memory growth on all GPUs
src/coastseg/zoo_model.py:        # if multiple GPUs are used use mirror strategy
src/coastseg/zoo_model.py:        if int(num_GPU) > 1:
src/coastseg/zoo_model.py:            "use_GPU": "0",
src/coastseg/zoo_model.py:                                         use_GPU:str="0",
src/coastseg/zoo_model.py:            use_GPU (str, optional): The GPU device to use. Defaults to "0".
src/coastseg/zoo_model.py:        use_GPU = settings.get('use_GPU', "0")
src/coastseg/zoo_model.py:            use_GPU=use_GPU,
src/coastseg/zoo_model.py:            use_GPU: str,
src/coastseg/zoo_model.py:                use_GPU (str): Whether to use GPU or not.
src/coastseg/zoo_model.py:            logger.info(f"use_GPU: {use_GPU}")
src/coastseg/zoo_model.py:                "use_GPU": use_GPU,
src/coastseg/models_UI.py:            "use_GPU": "0",

```
