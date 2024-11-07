# https://github.com/hfarias/mask_galaxy

```console
redes/two_class_zoo1/galaxia.py:    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
redes/two_class_zoo1/galaxia.py:    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
redes/two_class_zoo1/galaxia.py:    GPU_COUNT = 7
redes/two_class_zoo1/galaxia.py:    IMAGES_PER_GPU = 3
redes/two_class_zoo1/galaxia.py:    # We use a GPU with 12GB memory, which can fit two images.
redes/two_class_zoo1/galaxia.py:    # Adjust down if you use a smaller GPU.
redes/two_class_zoo1/galaxia.py:    #IMAGES_PER_GPU = 2
redes/two_class_zoo1/galaxia.py:            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
redes/two_class_zoo1/galaxia.py:            GPU_COUNT = 1
redes/two_class_zoo1/galaxia.py:            IMAGES_PER_GPU = 1
mrcnn/config.py:    # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
mrcnn/config.py:    GPU_COUNT = 1
mrcnn/config.py:    # Number of images to train with on each GPU. A 12GB GPU can typically
mrcnn/config.py:    # Adjust based on your GPU memory and image sizes. Use the highest
mrcnn/config.py:    # number that your GPU can handle for best performance.
mrcnn/config.py:    IMAGES_PER_GPU = 2
mrcnn/config.py:        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
mrcnn/model.py:                                   self.config.IMAGES_PER_GPU)
mrcnn/model.py:                                   self.config.IMAGES_PER_GPU)
mrcnn/model.py:                                    self.config.IMAGES_PER_GPU,
mrcnn/model.py:                                  self.config.IMAGES_PER_GPU,
mrcnn/model.py:                                  self.config.IMAGES_PER_GPU,
mrcnn/model.py:                                      self.config.IMAGES_PER_GPU)
mrcnn/model.py:            self.config.IMAGES_PER_GPU, names=names)
mrcnn/model.py:            self.config.IMAGES_PER_GPU)
mrcnn/model.py:                                   config.IMAGES_PER_GPU)
mrcnn/model.py:        # Add multi-GPU support.
mrcnn/model.py:        if config.GPU_COUNT > 1:
mrcnn/model.py:            model = ParallelModel(model, config.GPU_COUNT)
mrcnn/model.py:        the addition of multi-GPU support and the ability to exclude
mrcnn/model.py:        # In multi-GPU training, we wrap the model. Get layers
mrcnn/model.py:        # In multi-GPU training, we wrap the model. Get layers
mrcnn/parallel_model.py:Multi-GPU Support for Keras.
mrcnn/parallel_model.py:https://medium.com/@kuza55/transparent-multi-gpu-training-on-tensorflow-with-keras-8b0016fd9012
mrcnn/parallel_model.py:https://github.com/avolkov1/keras_experiments/blob/master/keras_exp/multigpu/
mrcnn/parallel_model.py:    """Subclasses the standard Keras Model and adds multi-GPU support.
mrcnn/parallel_model.py:    It works by creating a copy of the model on each GPU. Then it slices
mrcnn/parallel_model.py:    def __init__(self, keras_model, gpu_count):
mrcnn/parallel_model.py:        gpu_count: Number of GPUs. Must be > 1
mrcnn/parallel_model.py:        self.gpu_count = gpu_count
mrcnn/parallel_model.py:        the original model placed on different GPUs.
mrcnn/parallel_model.py:        # of the full inputs to all GPUs. Saves on bandwidth and memory.
mrcnn/parallel_model.py:        input_slices = {name: tf.split(x, self.gpu_count)
mrcnn/parallel_model.py:        # Run the model call() on each GPU to place the ops there
mrcnn/parallel_model.py:        for i in range(self.gpu_count):
mrcnn/parallel_model.py:            with tf.device('/gpu:%d' % i):
mrcnn/parallel_model.py:    # tries to run it on 2 GPUs. It saves the graph so it can be viewed
mrcnn/parallel_model.py:    GPU_COUNT = 2
mrcnn/parallel_model.py:    # Add multi-GPU support.
mrcnn/parallel_model.py:    model = ParallelModel(model, GPU_COUNT)

```
