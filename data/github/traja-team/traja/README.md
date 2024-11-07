# https://github.com/traja-team/traja

```console
docs/source/predictions.rst:    batch_size = 10 # How many sequences to train every step. Constrained by GPU memory.
traja/models/inference.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/models/train.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/models/train.py:        device: Selected device; 'cuda' or 'cpu'
traja/models/generative_models/vae.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/models/losses.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/models/predictive_models/ae.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/models/predictive_models/lstm.py:device = "cuda" if torch.cuda.is_available() else "cpu"
traja/plotting.py:    device = "cuda" if torch.cuda.is_available() else "cpu"
paper/paper.md:In comparison to mean-squared error loss, Huber loss is less sensitive to outliers in data: it is quadratic for small values of a, and linear for large values. It extends the PyTorch `SmoothL1Loss` class, where the $d$ parameter is set to 1.[^6] A common optimization algorithm is ADAM and is Traja’s default, but several others are provided as well. Although training with only a CPU is possible, a GPU can provide a $40-100x$ speedup [@Arpteg2018SoftwareEC].

```
