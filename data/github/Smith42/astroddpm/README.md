# https://github.com/Smith42/astroddpm

```console
denoising_diffusion_pytorch.py:DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
denoising_diffusion_pytorch.py:            x_start = torch.where(mask == True, torch.tensor(-1.0).to("cuda"), x_start)
infer.py:DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train.py:DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

```
