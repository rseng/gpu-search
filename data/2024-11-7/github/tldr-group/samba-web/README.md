# https://github.com/tldr-group/samba-web

```console
backend/forest_based.py:    """Given $feature_stack and $labels, flatten both and reshape accordingly. Add a class offset if using XGB gpu.
backend/forest_based.py:    if method == "gpu":
backend/encode.py:import torch.cuda as cuda
backend/encode.py:GPU_DESIRED = False
backend/encode.py:GPU_POSSIBLE = cuda.is_available()
backend/encode.py:GPU = GPU_DESIRED and GPU_POSSIBLE
backend/encode.py:if GPU:
backend/encode.py:    DEVICE = device("cuda:0")
backend/encode.py:    if GPU:
CONTRIBUTING.md:- **GPU featurisation using** [pyClesperanto](https://github.com/clEsperanto/pyclesperanto_prototype/): should be mostly a drag n drop replacement of multiscale_features so long as it returns an np array
CONTRIBUTING.md:- **A native app/GUI**: originally this project was a Python + Tkinter project, but that has largely fallen by the wayside. An updated Python version would be able to leverage all the GPU accelerations and ideally have a nice user interface without needing to install node/npm/yarn. It should be relatively easy to detach all the core logic from the server logic and wrap into a GUI.

```
