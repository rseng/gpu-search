# https://github.com/Music-and-Culture-Technology-Lab/omnizart

```console
Dockerfile:FROM tensorflow/tensorflow:2.5.0-gpu
CHANGELOG.md:- Upgrade Tensorflow version to 2.5.0 for Nvidia 30 series GPU compatibility.
omnizart/music/prediction.py:        Batch size for each step of prediction. The size is depending on the available GPU memory.
CONTRIBUTING.md:### For those who want to leverage the power of GPU for acceleration, make sure
CONTRIBUTING.md:### you have installed docker>=19.03 and the 'nvidia-container-toolkit' package.
CONTRIBUTING.md:# Execute the docker with GPU support
CONTRIBUTING.md:docker run --gpus all -it mctlab/omnizart:latest
cog.yaml:  gpu: true

```
