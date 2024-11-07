# https://github.com/wilsonrljr/sysidentpy

```console
docs/changelog/changelog.md:- MAJOR: Neural NARX now support CUDA
docs/changelog/changelog.md:    - Now the user can build Neural NARX models with CUDA support. Just add `device='cuda'` to use the GPU benefits.
docs/book/2 - NARMAX Model Representation.md:The user can use `cuda` following the same approach when build a neural network in **pytorch**
docs/book/2 - NARMAX Model Representation.md:torch.cuda.is_available()
docs/book/2 - NARMAX Model Representation.md:device = "cuda" if torch.cuda.is_available() else "cpu"
docs/book/2 - NARMAX Model Representation.md:The user have to pass the defined network to our NARXNN estimator and set `cuda` if available (or needed):
docs/book/2 - NARMAX Model Representation.md:if device == "cuda":
docs/book/2 - NARMAX Model Representation.md:    narx_net.net.to(torch.device("cuda"))
CHANGELOG.md:- MAJOR: Neural NARX now support CUDA
CHANGELOG.md:    - Now the user can build Neural NARX models with CUDA support. Just add `device='cuda'` to use the GPU benefits.
sysidentpy/neural_network/narx_nn.py:        self.device = self._check_cuda(device)
sysidentpy/neural_network/narx_nn.py:    def _check_cuda(self, device):
sysidentpy/neural_network/narx_nn.py:        if device not in ["cpu", "cuda"]:
sysidentpy/neural_network/narx_nn.py:            raise ValueError(f"device must be 'cpu' or 'cuda'. Got {device}")
sysidentpy/neural_network/narx_nn.py:        if device == "cuda":
sysidentpy/neural_network/narx_nn.py:            if torch.cuda.is_available():
sysidentpy/neural_network/narx_nn.py:                return torch.device("cuda")
sysidentpy/neural_network/narx_nn.py:                "No CUDA available. We set the device as CPU",

```
