# https://github.com/KennethEnevoldsen/augmenty

```console
docs/tutorials/training_using_augmenty/configs/default.cfg:gpu_allocator = null
docs/tutorials/training_using_augmenty/configs/default.cfg:gpu_allocator = ${system.gpu_allocator}
docs/tutorials/training_using_augmenty/project.yml:  gpu: -1
docs/tutorials/training_using_augmenty/project.yml:        --gpu-id ${vars.gpu} 
docs/tutorials/training_using_augmenty/project.yml:        --gpu-id ${vars.gpu}

```
