# https://github.com/Kaixhin/FGLab

```console
views/machine.jade:    dt.col-sm-3 GPUs
views/machine.jade:        - each gpu in machine.gpus
views/machine.jade:          li= gpu
api.raml:                "gpus": [],
api.raml:                "address": "http://gpu2.uni.ac.uk:5081",
api.raml:                "gpus": ["NVIDIA Corporation GK110B [GeForce GTX Titan Black] (rev a1)"],
api.raml:                "hostname": "gpu2",
api.raml:                "address": "http://gpu2.uni.ac.uk:5081",
api.raml:                "gpus": ["NVIDIA Corporation GK110B [GeForce GTX Titan Black] (rev a1)"],
api.raml:                "hostname": "gpu2",
api.raml:                "url": "http://gpu2.uni.ac.uk:5000/webhook",
examples/Recurrent-Attention-Model/recurrent-attention-model.json:  "cuda": {
examples/Recurrent-Attention-Model/README.md:For more information on Docker usage, including CUDA capabilities, please see the [source repo](https://github.com/Kaixhin/dockerfiles).

```
