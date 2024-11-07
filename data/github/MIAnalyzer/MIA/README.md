# https://github.com/MIAnalyzer/MIA

```console
mia/dl/machine_learning/segment_anything/sam.py:        self.device = "cuda"
mia/dl/machine_learning/segment_anything/automatic_mask_generator.py:            by the model. Higher numbers may be faster but use more GPU memory.
mia/ui/ui_Settings.py:        self.CBgpu = QComboBox(self.centralWidget)
mia/ui/ui_Settings.py:        self.CBgpu.setObjectName('GPUSettings')
mia/ui/ui_Settings.py:        self.CBgpu.addItem("default")
mia/ui/ui_Settings.py:            self.CBgpu.addItem(dev.name)                  
mia/ui/ui_Settings.py:        self.CBgpu.addItem("all gpus")
mia/ui/ui_Settings.py:        self.Lgpu = QLabel(self.centralWidget)
mia/ui/ui_Settings.py:        self.Lgpu.setText('GPU settings')
mia/ui/ui_Settings.py:        hlayout.addWidget(self.CBgpu)
mia/ui/ui_Settings.py:        hlayout.addWidget(self.Lgpu)
mia/ui/ui_Settings.py:        self.CBMixedPrecision.setToolTip('Select to use mixed precision during training, reduces gpu memory and increases computation speed on modern gpus')
mia/ui/ui_Settings.py:        self.CBgpu.currentIndexChanged.connect(self.setGPU)
mia/ui/ui_Settings.py:        self.CBgpu.setCurrentIndex(0)
mia/ui/ui_Settings.py:    def setGPU(self):
mia/ui/ui_Settings.py:        # finds cpu and gpu
mia/ui/ui_Settings.py:        gpus = tf.config.experimental.list_physical_devices('GPU') 
mia/ui/ui_Settings.py:        dev = self.CBgpu.currentIndex() - 1 
mia/ui/ui_Settings.py:                    tf.config.experimental.set_visible_devices(gpus, 'GPU')
mia/ui/ui_Settings.py:                    self.parent.PopupWarning('Use of multiple GPUs currently not supported')
mia/ui/ui_Settings.py:                    tf.config.experimental.set_visible_devices([], 'GPU')
mia/ui/ui_Settings.py:                    # set gpu
mia/ui/ui_Settings.py:                    tf.config.experimental.set_visible_devices(devices[dev], 'GPU')
mia/ui/ui_Settings.py:                # Visible devices must be set before GPUs have been initialized
mia/ui/ui_Settings.py:                self.parent.PopupWarning('GPU settings only take effect upon program start')
docs/source/gettingstarted/settings.rst:In the **GPU settings**, the engine for deep learning training and prediction can be set. Select the graphics processing unit (GPU), multiple GPUs or CPU that you want to use for deep learning. 
docs/source/gettingstarted/settings.rst:	Due to the high degree of parallelization deep learning training is strongly accelerated by using a GPU. See :doc:`../introduction/installation` to check if you have a compatible GPU.
docs/source/training/nntraining.rst:  Reduce batch size if (gpu-)memory is insufficient
README.md:In general, MIA should run on any system with Linux or windows. You can use the cpu only, but it is highly recommended to have a system with a [cuda-compatible](https://developer.nvidia.com/cuda-gpus) gpu (from NVIDIA) to accelerate neural network training.
mia_environment.yaml:  - nvidia
mia_environment.yaml:  - conda-forge::cudatoolkit=11.3.1
mia_environment.yaml:  - pytorch::pytorch-cuda=11.7
mia_environment.yaml:# or 'libcudart.x.x' not found and GPU not recognized,

```
