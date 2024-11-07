# https://github.com/jah1994/PyTorchDIA

```console
CITATION.md:  title={PyTorchDIA: a flexible, GPU-accelerated numerical approach to Difference Image Analysis},
README.md:PyTorchDIA is an open source, Pythonic, GPU-accelerated numerical DIA algorithm.
PyTorchDIA.py:A GPU accelerated approach for fast kernel (and differential background) solutions. The model image proposed in the Bramich (2008) algorithm is analogous to a very simple CNN, with a single convolutional layer / discrete pixel array (i.e. the kernel) and an added scalar bias (i.e. the differential background). We do not solve for the discrete pixel array directly in the linear least-squares sense. Rather, by making use of PyTorch tensors (GPU compatible multi-dimensional matrices) and neural network architecture, we solve via an efficient gradient-descent directed optimisation.
PyTorchDIA.py:# make sure to enable GPU acceleration (if availabel)!
PyTorchDIA.py:if torch.cuda.is_available() is True:
PyTorchDIA.py:  device = 'cuda'
PyTorchDIA.py:  print('CUDA not available, defaulting to CPU')
PyTorchDIA.py:      if torch.cuda.is_available() is True:
PyTorchDIA.py:    # Move model to GPU
PyTorchDIA.py:    if torch.cuda.is_available() is True:
PyTorchDIA.py:        # move L to cuda
PyTorchDIA.py:        if torch.cuda.is_available() is True:
PyTorchDIA.py:  #### Convert numpy images to tensors and move to GPU
PyTorchDIA.py:  # Move to GPU if CUDA available
PyTorchDIA.py:  if torch.cuda.is_available() is True:
PyTorchDIA.py:    print('Moving images to the GPU...')
PyTorchDIA.py:    time_to_move_to_GPU = time.time()
PyTorchDIA.py:    ## if providing a flat field move that to the GPU also
PyTorchDIA.py:  print("--- Time to move data onto GPU: %s ---" % (time.time() - time_to_move_to_GPU))

```
