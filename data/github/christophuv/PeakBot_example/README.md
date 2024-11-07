# https://github.com/christophuv/PeakBot_Example

```console
trainPB_MTBLS868.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
trainPB_MTBLS868.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_MTBLS868.py:import peakbot.train.cuda
trainPB_MTBLS868.py:    ## GPU information
trainPB_MTBLS868.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_MTBLS868.py:    ## Please consult the documentation of your GPU.
trainPB_MTBLS868.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_MTBLS868.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_MTBLS868.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_MTBLS868.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_MTBLS868.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_MTBLS868.py:                        peakbot.train.cuda.generateTestInstances(
generateMSMSExclusionList.py:## activate conda environment on jucuda
generateMSMSExclusionList.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
generateMSMSExclusionList.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
generateMSMSExclusionList.py:import peakbot.cuda            
generateMSMSExclusionList.py:    ## GPU information
generateMSMSExclusionList.py:    ## These values specify how the GPU is used for generating the training examples
generateMSMSExclusionList.py:    ## Please consult the documentation of your GPU.
generateMSMSExclusionList.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
generateMSMSExclusionList.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
generateMSMSExclusionList.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
generateMSMSExclusionList.py:                ### Detect local maxima with peak-like shapes## CUDA-GPU
generateMSMSExclusionList.py:                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
generateMSMSExclusionList.py:                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
generateMSMSExclusionList.py:                peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
trainPB_PHM.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
trainPB_PHM.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_PHM.py:import peakbot.train.cuda
trainPB_PHM.py:    ## GPU information
trainPB_PHM.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_PHM.py:    ## Please consult the documentation of your GPU.
trainPB_PHM.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_PHM.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_PHM.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_PHM.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_PHM.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_PHM.py:                        peakbot.train.cuda.generateTestInstances(
detectPB.py:## activate conda environment on jucuda
detectPB.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
detectPB.py:import tensorflow as tf## set memory limit on the GPU to 2GB to not run into problems with the pre- and post-processing steps
detectPB.py:    tf.config.experimental.list_physical_devices('GPU')[0],
detectPB.py:import peakbot.cuda            
detectPB.py:    ## GPU information
detectPB.py:    ## These values specify how the GPU is used for generating the training examples
detectPB.py:    ## Please consult the documentation of your GPU.
detectPB.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
detectPB.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
detectPB.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
detectPB.py:                ### Detect local maxima with peak-like shapes## CUDA-GPU
detectPB.py:                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
detectPB.py:                peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
quickExample_GPU.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
quickExample_GPU.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
quickExample_GPU.py:## Restrict TensorFlow to only allocate 1GB of memory on the first GPU
quickExample_GPU.py:gpus = tf.config.experimental.list_physical_devices('GPU')
quickExample_GPU.py:if gpus:
quickExample_GPU.py:        tf.config.experimental.set_virtual_device_configuration(gpus[0],
quickExample_GPU.py:        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
quickExample_GPU.py:        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
quickExample_GPU.py:        # Virtual devices must be set before GPUs have been initialized
quickExample_GPU.py:## GPU information
quickExample_GPU.py:## These values specify how the GPU is used for generating the training examples
quickExample_GPU.py:## Please consult the documentation of your GPU.
quickExample_GPU.py:## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64, exportBatchSize = 1024
quickExample_GPU.py:## Values for a high-end HPC Nvidia Tesla V100S card with 32GB GPU-memory are blockdim = 16, griddim = 512, exportBatchSize = 12288
quickExample_GPU.py:## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
quickExample_GPU.py:strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
quickExample_GPU.py:import peakbot.train.cuda
quickExample_GPU.py:                        peakbot.train.cuda.generateTestInstances(
quickExample_GPU.py:            ### Detect local maxima with peak-like shapes## CUDA-GPU
quickExample_GPU.py:            peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
quickExample_GPU.py:            peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
README.md:This script shows how to train a new PeakBot model. It generates a training dataset (T) and 4 validation datasets (V, iT, iV, eV) from different LC-HRMS chromatograms and different reference feature and background lists automatically. Then using the computer's GPU the new PeakBot model is trained and evaluated. 
README.md:The main functions to generate training instances from a LC-HRMS chromatogram and a reference peak and background list are the functions `peakbot.train.cuda.generateTestInstances` for generating a large set of training instances and `peakbot.trainPeakBotModel` for training a new PeakBot model with the previously generated training instances.
README.md:The main functions to detect chromatographic peaks in a LC-HRMS chromatogram with a PeakBot model are `peakbot.cuda.preProcessChromatogram` for extracting the standardized areas from the chromatogram and `peakbot.runPeakBot` for testing the standardized area for chromatographic peaks or backgrounds. 
README.md:The main functions to group detected features from several samples are `peakbot.cuda.KNNalignFeatures` for aligning the detected features using a k-nearest-neighbor approach and `peakbot.cuda.groupFeatures` for calculating the groups. 
quickFindCUDAParameters.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
quickFindCUDAParameters.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
quickFindCUDAParameters.py:## GPU information
quickFindCUDAParameters.py:## These values specify how the GPU is used for generating the training examples
quickFindCUDAParameters.py:## Please consult the documentation of your GPU.
quickFindCUDAParameters.py:## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
quickFindCUDAParameters.py:## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
quickFindCUDAParameters.py:strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
quickFindCUDAParameters.py:import peakbot.train.cuda
quickFindCUDAParameters.py:                                    ### Detect local maxima with peak-like shapes## CUDA-GPU
quickFindCUDAParameters.py:                                    peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
quickFindCUDAParameters.py:                                    peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
trainPB_MTBLS797.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
trainPB_MTBLS797.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_MTBLS797.py:import peakbot.train.cuda
trainPB_MTBLS797.py:    ## GPU information
trainPB_MTBLS797.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_MTBLS797.py:    ## Please consult the documentation of your GPU.
trainPB_MTBLS797.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_MTBLS797.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_MTBLS797.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_MTBLS797.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_MTBLS797.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_MTBLS797.py:                        peakbot.train.cuda.generateTestInstances(
group.py:## activate conda environment on jucuda
group.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
group.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
group.py:import peakbot.cuda
group.py:    ## GPU information
group.py:    ## These values specify how the GPU is used for generating the training examples
group.py:    ## Please consult the documentation of your GPU.
group.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
group.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
group.py:            features = peakbot.cuda.KNNalignFeatures(features, featuresOri, 
group.py:            headers, features = peakbot.cuda.groupFeatures(features, featuresOri, expParams[expName]["rtMaxDiffGrouping"], expParams[expName]["ppmMaxDiffGrouping"], fileMapping, blockdim = blockdim, griddim = griddim)
trainPB_ST001450.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainPB_ST001450.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_ST001450.py:import peakbot.train.cuda
trainPB_ST001450.py:    ## GPU information
trainPB_ST001450.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_ST001450.py:    ## Please consult the documentation of your GPU.
trainPB_ST001450.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_ST001450.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_ST001450.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_ST001450.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_ST001450.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_ST001450.py:                        peakbot.train.cuda.generateTestInstances(
trainPB_MTBLS1358.py:os.environ["CUDA_VISIBLE_DEVICES"]="0"
trainPB_MTBLS1358.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_MTBLS1358.py:import peakbot.train.cuda
trainPB_MTBLS1358.py:    ## GPU information
trainPB_MTBLS1358.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_MTBLS1358.py:    ## Please consult the documentation of your GPU.
trainPB_MTBLS1358.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_MTBLS1358.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_MTBLS1358.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_MTBLS1358.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_MTBLS1358.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_MTBLS1358.py:                        peakbot.train.cuda.generateTestInstances(
quickExample_CPU.py:os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
quickExample_CPU.py:## GPU information
quickExample_CPU.py:## These values specify how the GPU is used for generating the training examples
quickExample_CPU.py:## Please consult the documentation of your GPU.
quickExample_CPU.py:## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64, exportBatchSize = 1024
quickExample_CPU.py:## Values for a high-end HPC Nvidia Tesla V100S card with 32GB GPU-memory are blockdim = 16, griddim = 512, exportBatchSize = 12288
quickExample_CPU.py:## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
quickExample_CPU.py:            ### Detect local maxima with peak-like shapes## CUDA-GPU
trainPB_WheatEar.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
trainPB_WheatEar.py:tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
trainPB_WheatEar.py:import peakbot.train.cuda
trainPB_WheatEar.py:    ## GPU information
trainPB_WheatEar.py:    ## These values specify how the GPU is used for generating the training examples
trainPB_WheatEar.py:    ## Please consult the documentation of your GPU.
trainPB_WheatEar.py:    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
trainPB_WheatEar.py:    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
trainPB_WheatEar.py:    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
trainPB_WheatEar.py:    examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
trainPB_WheatEar.py:    logDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/PeakBot/logs/"
trainPB_WheatEar.py:                        peakbot.train.cuda.generateTestInstances(

```
