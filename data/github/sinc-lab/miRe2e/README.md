# https://github.com/sinc-lab/miRe2e

```console
README.md:A Python>=3.7 distribution is required to use this package. If you plan to use a GPU, please check the [pytorch web](https://pytorch.org/get-started/locally/) to configure it correctly before installing this package. 
README.md:By default, the package use the cpu, thus it is recommended to use GPU to train the models.
README.md:By default, MiRe2e object is created to use cpu only. If the server has at least one GPU, you need to pass the parameter device='cuda'. You can select a specific device, i.e. device='cuda:1' if you to have two  GPUs and you want to use the second one:
README.md:model = MiRe2e(device='cuda:1')
README.md:Training the models may take several hours and requires GPU processing 
README.md:Training scripts were made for a 12GB GPU. You can adjust batch_size according to your hardware setup.
miRe2e/mire2e.py:        device: Either "cpu" or "cuda"

```
