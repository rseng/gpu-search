# https://github.com/tsipkens/atems

```console
main_carboseg.m:py_exec = 'C:\Users\tsipk\anaconda3\envs\carboseg-gpu\python.exe';
carboseg/environment-gpu.yml:name: carboseg-gpu
carboseg/environment-gpu.yml:- cudatoolkit
carboseg/environment-gpu.yml:    - onnxruntime-gpu
carboseg/README.md:Alternatively, one can set up an environment to take advantage of a GPU. The example here applied to CUDA-enabled scenerios. In this case, one can create a **carboseg-gpu** environmnet using:
carboseg/README.md:conda env create --file environment-gpu.yml
carboseg/README.md:> NOTE: If CUDA is available and the batch of images is reasonably large, it may be faster to save the images and run the classification on a GPU in Python directly (again, see the next subsection for this option). 

```
