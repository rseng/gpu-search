# https://github.com/deepmedic/deepmedic

```console
documentation/README.md:  * [1.3. GPU Processing](#14-gpu-processing)
documentation/README.md:  * [2.2 Running it on a GPU](#22-running-it-on-a-gpu)
documentation/README.md:By consulting the previous link, ensure that your system has **CUDA** version and **cuDNN** versions compatible with the tensorflow version you are installing.
documentation/README.md:$ pip install tensorflow-gpu==2.6.2
documentation/README.md:specific cudnn versions (see TF docs). We need Cudnn that is compatible with TF and your system's Nvidia drivers.
documentation/README.md:#### 1.3. GPU Processing
documentation/README.md:#### Install CUDA: (Deprecated)
documentation/README.md: of the required libraries. As long as you have installed GPU drivers, cudnn tends to install the rest. 
documentation/README.md:Small networks can be run on the cpu. But 3D CNNs of considerable size require processing on the GPU. For this, an installation of [Nvidia’s CUDA](https://developer.nvidia.com/cuda-toolkit) is
documentation/README.md: needed. Make sure to acquire a version compatible with your GPU drivers. TensorFlow needs to be able to find CUDA’s compiler, the **nvcc**, in the environment’s path. It also dynamically links to **cublas.so** libraries, which need to be visible in the environment’s.
documentation/README.md:Prior to running DeepMedic on the GPU, you must manually add the paths to the folders containing these files in your environment's variables. As an example in a *bash* shell:
documentation/README.md:$ export CUDA_HOME=/path/to/cuda                   # If using cshell instead of bash: setenv CUDA_HOME /path/to/cuda
documentation/README.md:$ export LD_LIBRARY_PATH=/path/to/cuda/lib64
documentation/README.md:$ export PATH=/path/to/cuda/bin:$PATH
documentation/README.md:#### 2.2 Running it on a GPU
documentation/README.md:Now lets check the important part... If using the **DeepMedic on the GPU** is alright on your system. First, delete the `examples/output/` folder for a clean start. Now, most importantly, place the path to **CUDA**'s *nvcc* into your *PATH* and to the *cublas.so* in your *LD_LIBRARY_PATH* (see [section 1.3](#13-gpu-processing))
documentation/README.md:You need to perform the steps we did before for training and testing with a model, but on the GPU. To do this, repeat the previous commands and pass the additional option `-dev cuda`. For example: 
documentation/README.md:               -dev cuda0
documentation/README.md:You can replace 0 to specify another device number, if your machine has multiple GPUs. The processes should result in similar outputs as before. **Make sure the process runs on the GPU**, by running the command `nvidia-smi`. You should see your python process assigned to the specified GPU. If all processes finish as normal and you get no errors, amazing. **Now it seems that really everything works :)** Continue to the next section and find more details about the DeepMedic and how to use the large version of our network!
documentation/README.md:**Possible problems with the GPU**: If TensorFlow does not find correct versions for **CUDA** and **cuDNN** (depends on TensorFlow version), it will fall back to the CPU version by default. If this happens, right after the model creation and before the main training process starts, some warnings will be thrown by TensorFlow, along the lines below:
documentation/README.md:2018-06-06 14:39:35.676554: E tensorflow/stream_executor/cuda/cuda_driver.cc:406] failed call to cuInit: CUDA_ERROR_NO_DEVICE
documentation/README.md:2018-06-06 14:39:35.676616: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: neuralmedic.doc.ic.ac.uk
documentation/README.md:2018-06-06 14:39:35.676626: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: neuralmedic.doc.ic.ac.uk
documentation/README.md:2018-06-06 14:39:35.676664: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: 384.111.0
documentation/README.md:2018-06-06 14:39:35.676699: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: 384.111.0
documentation/README.md:2018-06-06 14:39:35.676708: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:300] kernel version seems to match DSO: 384.111.0
documentation/README.md:If the process does not start on the GPU as required, please ensure you have *CUDA* and *cuDNN* versions that are compatible with the TF version you have (https://www.tensorflow.org/install), and that you environment variables are correctly setup. See Section 1.4 about some pointers, and the *CUDA* website.
documentation/README.md:- Batch Size : The number of segments to process simultaneously on GPU. In training, bigger batch sizes achieve better convergence and results, but require more computation and memory. Batch sizes for Validation and Inference are less important, greater once just speedup the process.
documentation/README.md:               -dev cuda0
documentation/README.md:Note that you can change 0 with another number of a GPU device, if your machine has **multiple GPUs**.
documentation/README.md:               -dev cuda0
documentation/README.md:               -dev cuda0
documentation/README.md:- numberTrainingSegmentsLoadedOnGpuPerSubep: At every subepoch, we extract in total this many segments, which are loaded on the GPU in order to perform the optimization steps. Number of optimization steps per subepoch is this number divided by the batch-size-training (see model-config). The more segments, the more GPU memory and computation required.
documentation/README.md:- batchsize_train: Size of a training batch. The bigger, the more gpu-memory is required.
documentation/README.md:- num_processes_sampling: Samples needed for next validation/train can be extracted in parallel while performing current train/validation on GPU. Specify number of parallel sampling processes.
documentation/README.md:- numberValidationSegmentsLoadedOnGpuPerSubep: on how many validation segments (samples) to perform the validation.
documentation/README.md:               -dev cuda0
documentation/README.md:The provided configuration of the DeepMedic takes roughly 2 days to get trained on an NVIDIA GTX Titan X. Inference on a standard size brain scan should take 2-3 minutes. Adjust configuration of training and testing or consider downsampling your data if it takes much longer for your task.
deepMedicRun:ARG_GPU_PROC = "cuda"
deepMedicRun:    parser.add_argument(OPT_DEVICE, default = DEF_DEV_PROC, dest='device', type=str,  help="Specify the device to run the process on. Values: [" + ARG_CPU_PROC + "] or [" + ARG_GPU_PROC + "] (default = " + DEF_DEV_PROC + ").\n"+\
deepMedicRun:                                                                    "In the case of multiple GPUs, specify a particular GPU device with a number, in the format: " + OPT_DEVICE + " " + ARG_GPU_PROC + "0 \n"+\
deepMedicRun:                                                                    "NOTE: For GPU processing, CUDA libraries must be first added in your environment's PATH and LD_LIBRARY_PATH. See accompanying documentation.")
deepMedicRun:    if devArg == ARG_GPU_PROC: return
deepMedicRun:    if devArg.startswith(ARG_GPU_PROC) and str_is_int(devArg[len(ARG_GPU_PROC):]): return
deepMedicRun:            "\tValues: [" + ARG_CPU_PROC + "] or [" + ARG_GPU_PROC + "] (Default = " + DEF_DEV_PROC + ").\n"+\
deepMedicRun:            "\tIn the case of multiple GPUs, specify a particular GPU device with a number, in the format: " + ARG_GPU_PROC + "2. Exiting.")
deepMedicRun:    # Setup cpu / gpu devices.
deepMedicRun:        os.environ["CUDA_VISIBLE_DEVICES"] = ""
deepMedicRun:    elif dev_string == ARG_GPU_PROC:
deepMedicRun:        sess_device = None # With None, TF will get all cuda devices and assign to the first.
deepMedicRun:    if dev_string.startswith(ARG_GPU_PROC) and str_is_int(dev_string[ len(ARG_GPU_PROC):]):
deepMedicRun:        os.environ["CUDA_VISIBLE_DEVICES"] = dev_string[len(ARG_GPU_PROC):]
deepMedicRun:        sess_device = "/device:GPU:0"
deepmedic/neuralnet/trainer.py:        self._num_epochs_trained_tfv = tf.Variable(0, dtype="int64", trainable=False, name="num_epochs_trained") # int32 tf.vars cannot be (explicitly) loaded to gpu.
deepmedic/routines/training.py:    # This created problems in the GPU when cnmem is used. Not sure this is needed with Tensorflow. Probably.
deepmedic/frontEnd/trainSession.py:            # Explicit device assignment, throws an error if GPU is specified but not available.
deepmedic/frontEnd/trainSession.py:                                              device_count={'CPU': 999, 'GPU': 99})) as sessionTf:
deepmedic/frontEnd/testSession.py:            with graphTf.device(sess_device): # Throws an error if GPU is specified but not available.
deepmedic/frontEnd/testSession.py:        with tf.compat.v1.Session(graph=graphTf, config=tf.compat.v1.ConfigProto(log_device_placement=False, device_count={'CPU':999, 'GPU':99})) as sessionTf:
deepmedic/frontEnd/configParsing/trainConfig.py:    NUM_TR_SEGMS_LOADED_PERSUB = "numberTrainingSegmentsLoadedOnGpuPerSubep"
deepmedic/frontEnd/configParsing/trainConfig.py:    NUM_VAL_SEGMS_LOADED_PERSUB = "numberValidationSegmentsLoadedOnGpuPerSubep"  # For val on samples.
deepmedic/dataManagement/sampling.py:# Main sampling process during training. Executed in parallel while training on a batch on GPU.
deepmedic/dataManagement/sampling.py:                           max_subjects_on_gpu_for_subepoch,
deepmedic/dataManagement/sampling.py:                           get_max_subjects_for_gpu_even_if_total_less=False):
deepmedic/dataManagement/sampling.py:    if max_subjects_on_gpu_for_subepoch >= n_total_subjects:
deepmedic/dataManagement/sampling.py:        # This is if I want to have a certain amount on GPU, even if total subjects are less.
deepmedic/dataManagement/sampling.py:        if get_max_subjects_for_gpu_even_if_total_less:
deepmedic/dataManagement/sampling.py:            while len(random_order_chosen_subjects) < max_subjects_on_gpu_for_subepoch:
deepmedic/dataManagement/sampling.py:                number_of_extra_subjects_to_get_to_fill_gpu = min(
deepmedic/dataManagement/sampling.py:                    max_subjects_on_gpu_for_subepoch - len(random_order_chosen_subjects), n_total_subjects)
deepmedic/dataManagement/sampling.py:                random_order_chosen_subjects += (subjects_indices[:number_of_extra_subjects_to_get_to_fill_gpu])
deepmedic/dataManagement/sampling.py:            assert len(random_order_chosen_subjects) == max_subjects_on_gpu_for_subepoch
deepmedic/dataManagement/sampling.py:        random_order_chosen_subjects += subjects_indices[:max_subjects_on_gpu_for_subepoch]
README.md:  * [1.3. GPU Processing](#14-gpu-processing)
README.md:  * [2.2 Running it on a GPU](#22-running-it-on-a-gpu)
README.md:By consulting the previous link, ensure that your system has **CUDA** version and **cuDNN** versions compatible with the tensorflow version you are installing.
README.md:$ pip install tensorflow-gpu==2.6.2
README.md:specific cudnn versions (see TF docs). We need Cudnn that is compatible with TF and your system's Nvidia drivers.
README.md:#### 1.3. GPU Processing
README.md:#### Install CUDA: (Deprecated)
README.md: of the required libraries. As long as you have installed GPU drivers, cudnn tends to install the rest. 
README.md:Small networks can be run on the cpu. But 3D CNNs of considerable size require processing on the GPU. For this, an installation of [Nvidia’s CUDA](https://developer.nvidia.com/cuda-toolkit) is
README.md: needed. Make sure to acquire a version compatible with your GPU drivers. TensorFlow needs to be able to find CUDA’s compiler, the **nvcc**, in the environment’s path. It also dynamically links to **cublas.so** libraries, which need to be visible in the environment’s.
README.md:Prior to running DeepMedic on the GPU, you must manually add the paths to the folders containing these files in your environment's variables. As an example in a *bash* shell:
README.md:$ export CUDA_HOME=/path/to/cuda                   # If using cshell instead of bash: setenv CUDA_HOME /path/to/cuda
README.md:$ export LD_LIBRARY_PATH=/path/to/cuda/lib64
README.md:$ export PATH=/path/to/cuda/bin:$PATH
README.md:#### 2.2 Running it on a GPU
README.md:Now lets check the important part... If using the **DeepMedic on the GPU** is alright on your system. First, delete the `examples/output/` folder for a clean start. Now, most importantly, place the path to **CUDA**'s *nvcc* into your *PATH* and to the *cublas.so* in your *LD_LIBRARY_PATH* (see [section 1.3](#13-gpu-processing))
README.md:You need to perform the steps we did before for training and testing with a model, but on the GPU. To do this, repeat the previous commands and pass the additional option `-dev cuda`. For example: 
README.md:               -dev cuda0
README.md:You can replace 0 to specify another device number, if your machine has multiple GPUs. The processes should result in similar outputs as before. **Make sure the process runs on the GPU**, by running the command `nvidia-smi`. You should see your python process assigned to the specified GPU. If all processes finish as normal and you get no errors, amazing. **Now it seems that really everything works :)** Continue to the next section and find more details about the DeepMedic and how to use the large version of our network!
README.md:**Possible problems with the GPU**: If TensorFlow does not find correct versions for **CUDA** and **cuDNN** (depends on TensorFlow version), it will fall back to the CPU version by default. If this happens, right after the model creation and before the main training process starts, some warnings will be thrown by TensorFlow, along the lines below:
README.md:2018-06-06 14:39:35.676554: E tensorflow/stream_executor/cuda/cuda_driver.cc:406] failed call to cuInit: CUDA_ERROR_NO_DEVICE
README.md:2018-06-06 14:39:35.676616: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:158] retrieving CUDA diagnostic information for host: neuralmedic.doc.ic.ac.uk
README.md:2018-06-06 14:39:35.676626: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:165] hostname: neuralmedic.doc.ic.ac.uk
README.md:2018-06-06 14:39:35.676664: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:189] libcuda reported version is: 384.111.0
README.md:2018-06-06 14:39:35.676699: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:193] kernel reported version is: 384.111.0
README.md:2018-06-06 14:39:35.676708: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:300] kernel version seems to match DSO: 384.111.0
README.md:If the process does not start on the GPU as required, please ensure you have *CUDA* and *cuDNN* versions that are compatible with the TF version you have (https://www.tensorflow.org/install), and that you environment variables are correctly setup. See Section 1.4 about some pointers, and the *CUDA* website.
README.md:- Batch Size : The number of segments to process simultaneously on GPU. In training, bigger batch sizes achieve better convergence and results, but require more computation and memory. Batch sizes for Validation and Inference are less important, greater once just speedup the process.
README.md:               -dev cuda0
README.md:Note that you can change 0 with another number of a GPU device, if your machine has **multiple GPUs**.
README.md:               -dev cuda0
README.md:               -dev cuda0
README.md:- numberTrainingSegmentsLoadedOnGpuPerSubep: At every subepoch, we extract in total this many segments, which are loaded on the GPU in order to perform the optimization steps. Number of optimization steps per subepoch is this number divided by the batch-size-training (see model-config). The more segments, the more GPU memory and computation required.
README.md:- batchsize_train: Size of a training batch. The bigger, the more gpu-memory is required.
README.md:- num_processes_sampling: Samples needed for next validation/train can be extracted in parallel while performing current train/validation on GPU. Specify number of parallel sampling processes.
README.md:- numberValidationSegmentsLoadedOnGpuPerSubep: on how many validation segments (samples) to perform the validation.
README.md:               -dev cuda0
README.md:The provided configuration of the DeepMedic takes roughly 2 days to get trained on an NVIDIA GTX Titan X. Inference on a standard size brain scan should take 2-3 minutes. Adjust configuration of training and testing or consider downsampling your data if it takes much longer for your task.
examples/configFiles/tinyCnn/model/modelConfig.cfg:#  [Optional] Bigger image segments for Inference are safe to use and only speed up the process. Only limitation is the GPU memory.
examples/configFiles/tinyCnn/train/trainConfig.cfg:#  Every subepoch, extract in total this many segments and load them on the GPU. Memory Limitated. Default: 1000
examples/configFiles/tinyCnn/train/trainConfig.cfg:#  Note: This number in combination with the batchsize define the number of optimization steps per subepoch (=NumOfSegmentsOnGpu / BatchSize).
examples/configFiles/tinyCnn/train/trainConfig.cfg:numberTrainingSegmentsLoadedOnGpuPerSubep = 1000
examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg:#  Every subepoch, extract in total this many segments and load them on the GPU. Memory Limitated. Default: 1000
examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg:#  Note: This number in combination with the batchsize define the number of optimization steps per subepoch (=NumOfSegmentsOnGpu / BatchSize).
examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg:numberTrainingSegmentsLoadedOnGpuPerSubep = 1000
examples/configFiles/tinyCnn/train/trainConfigWithValidation.cfg:numberValidationSegmentsLoadedOnGpuPerSubep = 5000
examples/configFiles/deepMedic/model/modelConfig_wide1.cfg:#  [Optional] Bigger image segments for Inference are safe to use and only speed up the process. Only limitation is the GPU memory.
examples/configFiles/deepMedic/model/modelConfig.cfg:#  [Optional] Bigger image segments for Inference are safe to use and only speed up the process. Only limitation is the GPU memory.
examples/configFiles/deepMedic/train/trainConfig.cfg:#  Every subepoch, extract in total this many segments and load them on the GPU. Memory Limitated. Default: 1000
examples/configFiles/deepMedic/train/trainConfig.cfg:#  Note: This number in combination with the batchsize define the number of optimization steps per subepoch (=NumOfSegmentsOnGpu / BatchSize).
examples/configFiles/deepMedic/train/trainConfig.cfg:numberTrainingSegmentsLoadedOnGpuPerSubep = 1000
examples/configFiles/deepMedic/train/trainConfig.cfg:numberValidationSegmentsLoadedOnGpuPerSubep = 5000

```
