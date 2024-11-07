# https://github.com/automl/amltk

```console
examples/pytorch-example.py:    device: str = "cpu",  # Change if you have a GPU
examples/pytorch-example.py:    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
src/amltk/pipeline/xgboost.py:    if tree_method == "hist" and ("cuda" in device or "gpu" in device):

```
