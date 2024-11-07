# https://github.com/SERVIR/servir-aces

```console
docs/usage.md:***Note:*** this takes a while to run and needs access to the GPU to run it faster. If you don't have access to GPU or don't want to wait while its running, we provide you with the trained model via the Google Cloud Storage. After you get the data via `gsutil` above, you will see a folder called `models` inside it. For the U-Net, you can download the `unet_v1` folder and put it inside the `MODEL_DIR`, and for DNN, you can download the `dnn_v1` folder and place it inside the `MODEL_DIR`.
docs/index.md:***Note: We have a Jupyter Notebook Colab example that you can use to run without having to worry too much about setting up locally. You can find the relevant notebook [here](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb). Note, however, the resources especially GPU may not be fully available via Colab for unpaid version***
workflow/v2/host_vertex_ai.py:# --accelerator=type=nvidia-tesla-k80,count=1"""
README.md:***Note: We have a Jupyter Notebook Colab example that you can use to run without having to worry too much about setting up locally. You can find the relevant notebook [here](https://github.com/SERVIR/servir-aces/blob/main/notebook/Rice_Mapping_Bhutan_2021.ipynb). Note, however, the resources especially GPU may not be fully available via Colab for unpaid version***
aces/utils.py:        Configure TensorFlow to allocate GPU memory dynamically.
aces/utils.py:        If GPUs are found, this method enables memory growth for each GPU.
aces/utils.py:        physical_devices = tf.config.list_physical_devices("GPU")
aces/utils.py:            print(f" > Found {len(physical_devices)} GPUs")
aces/utils.py:            print(" > No GPUs found")

```
