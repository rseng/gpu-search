# https://github.com/tomasplsek/CADET

```console
pycadet/pycadet.py:# Configure GPU
pycadet/pycadet.py:def configure_GPU():
pycadet/pycadet.py:    # # DISABLE GPU
pycadet/pycadet.py:    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
pycadet/pycadet.py:    # CONFIGURATION FOR DEDICATED NVIDIA GPU 
pycadet/pycadet.py:    # gpus = list_physical_devices('GPU')
pycadet/pycadet.py:    # if len(gpus) > 0:
pycadet/pycadet.py:    #         set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=1000)])
pycadet/pycadet.py:    #         print(f"\n{len(gpus)} GPUs detected. Configuring the memory limit to 1GB.")
pycadet/pycadet.py:    #         print(f"\n{len(gpus)} GPUs detected. Using default GPU settings.")
pycadet/pycadet.py:    # else: print("\nNo GPUs detected. Using a CPU.")
pycadet/pycadet.py:    gpus = list_physical_devices('GPU')
pycadet/pycadet.py:    if len(gpus) > 0: print(f"{len(gpus)} GPUs detected.\n")
pycadet/pycadet.py:    else: print("No GPUs detected. Using a CPU.\n")
pycadet/pycadet.py:# configure_GPU()
training_testing/test_CADET.py:gpus = list_physical_devices('GPU')
training_testing/test_CADET.py:set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2000)])
training_testing/test_CADET.py:print(len(gpus), "Physical GPUs")
training_testing/test_CADET.py:# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
training_testing/train_CADET.py:# GPU initialization
training_testing/train_CADET.py:gpus = list_physical_devices('GPU')
training_testing/train_CADET.py:set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=5000)])
training_testing/train_CADET.py:print(len(gpus), "Physical GPUs")
training_testing/train_CADET.py:# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
training_testing/train_CADET.py:# # GPU initialization
training_testing/train_CADET.py:# gpus = list_physical_devices('GPU')
training_testing/train_CADET.py:# set_virtual_device_configuration(gpus[0], [VirtualDeviceConfiguration(memory_limit=2600)])
training_testing/train_CADET.py:# print(len(gpus), "Physical GPUs")
README.md:For Conda environments, it is recommended to install the dependencies beforehand as some of the packages can be tricky to install in an existing environment (especially `tensorflow`) and on some machines (especially new Macs). For machines with dedicated NVIDIA GPUs, `tensorflow-gpu` can be installed to allow the CADET model to leverage the GPU for faster inference.

```
