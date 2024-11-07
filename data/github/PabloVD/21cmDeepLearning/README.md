# https://github.com/PabloVD/21cmDeepLearning

```console
Saliency_astro.py:    if train_on_gpu:
Saliency_astro.py:        Xmod = Xmod.cuda()
Saliency_astro.py:        model.cuda()
HI2Astro.py:    if train_on_gpu:
HI2Astro.py:        my_dict = torch.load(best_Unet_model,map_location=torch.device('cuda'))
HI2Astro.py:if train_on_gpu:
HI2Astro.py:    astromodel.cuda()
Source/functions.py:# Use CUDA GPUs if available
Source/functions.py:train_on_gpu = torch.cuda.is_available()
Source/functions.py:if train_on_gpu:
Source/functions.py:    print('\nCUDA is available! Training on GPU.')
Source/functions.py:    device = torch.device('cuda')
Source/functions.py:    print('\nCUDA is not available. Training on CPU.')
Source/functions.py:            if train_on_gpu:
Source/functions.py:                input, target = input.cuda(), target.cuda()
Source/functions.py:                if train_on_gpu:
Source/functions.py:                    input, target = input.cuda(), target.cuda()
Source/functions.py:            if train_on_gpu:
Source/functions.py:                input, target = input.cuda(), target.cuda()
HI2DM.py:if train_on_gpu:
HI2DM.py:    model.cuda()

```
