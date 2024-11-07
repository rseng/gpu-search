# https://github.com/nlesc-nano/swan

```console
docs/tutorial_models.rst:  use_cuda: True
CHANGELOG.md:* Allow to  train in **GPU**.
tests/test_se3_transformer.py:    modeller = TorchModeller(net, DATA, use_cuda=False, replace_state=True)
tests/test_se3_transformer.py:    researcher = TorchModeller(net, DATA, use_cuda=False)
tests/files/input_test_fingerprint_predict.yml:use_cuda: False
tests/files/input_test_graph_geometries.yml:use_cuda: False
tests/files/input_test_graph_train.yml:use_cuda: False
tests/files/input_test_graph_predict.yml:use_cuda: False
tests/files/input_test_e3nn_train.yml:use_cuda: True
tests/files/input_test_fingerprint_train.yml:use_cuda: False
tests/test_gpytorch.py:    researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
tests/test_gpytorch.py:    researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
swan/modeller/gp_modeller.py:            replace_state: bool = False, use_cuda: bool = False):
swan/modeller/gp_modeller.py:        use_cuda
swan/modeller/gp_modeller.py:            Train the model using Cuda
swan/modeller/gp_modeller.py:            network, data, replace_state=replace_state, use_cuda=use_cuda)
swan/modeller/torch_modeller.py:                 use_cuda: bool = False):
swan/modeller/torch_modeller.py:        use_cuda
swan/modeller/torch_modeller.py:            Train the model using Cuda
swan/modeller/torch_modeller.py:        # cuda support
swan/modeller/torch_modeller.py:        self.use_cuda = use_cuda
swan/modeller/torch_modeller.py:        if self.use_cuda:
swan/modeller/torch_modeller.py:            self.device = torch.device("cuda")
scripts/predict_torch.py:    researcher = Modeller(net, data, use_cuda=False)
scripts/predict_gp.py:researcher = GPModeller(model, data, use_cuda=False, replace_state=False)
scripts/run_gp_models.py:researcher = GPModeller(model, data, use_cuda=False, replace_state=True)
scripts/run_torch_models.py:researcher = TorchModeller(net, data, use_cuda=False)
scripts/predict_all.py:    researcher = Modeller(net, data, use_cuda=False)

```
