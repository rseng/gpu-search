# https://github.com/jet-net/JetNet

```console
jetnet/losses/losses.py:        device (str): 'cpu' or 'cuda'. Defaults to 'cpu'.
jetnet/losses/losses.py:        assert device == "cpu" or device == "cuda", "invalid device type"
jetnet/evaluation/gen_metrics.py:        device (str): 'cpu' or 'cuda'. If not specified, defaults to cuda if available else cpu.
jetnet/evaluation/gen_metrics.py:        device = "cuda" if torch.cuda.is_available() else "cpu"
jetnet/evaluation/gen_metrics.py:    assert device == "cuda" or device == "cpu", "Invalid device type"

```
