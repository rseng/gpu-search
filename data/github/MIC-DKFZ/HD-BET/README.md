# https://github.com/MIC-DKFZ/HD-BET

```console
HD_BET/run.py:        net.cuda(device)
HD_BET/predict_case.py:            a = a.cuda(main_device)
HD_BET/hd_bet_cli.py:                                                               'Must be either int or str. Use int for GPU id or '
readme.md:- HD-BET is very fast on GPU with <10s run time per MRI sequence. Even on CPU it
readme.md:as GPU support. Running on GPU is a lot faster though and should always be
readme.md:By default, HD-BET will run in GPU mode, use the parameters of all five models
readme.md:### GPU is nice, but I don't have one of those... What now?
readme.md:1. **How much GPU memory do I need to run HD-BET?** We ran all our experiments
readme.md:   on NVIDIA Titan X GPUs with 12 GB memory. For inference you will need less,
readme.md:   image should run with less than 4 GB of GPU memory consumption. If you run
readme.md:3. **What run time can I expect on CPU/GPU?** This depends on your MRI image
readme.md:   included) are just a couple of seconds for GPU and about 2 Minutes on CPU

```
