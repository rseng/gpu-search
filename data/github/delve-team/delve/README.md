# https://github.com/delve-team/delve

```console
README.rst:   from torch.cuda import is_available
README.rst:     device = "cuda:0" if is_available() else "cpu"
README.rst:         with torch.cuda.amp.autocast():
docs/source/gallery/extract-saturation.py:device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
docs/source/gallery/extract-saturation.rst:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
docs/examples/extract-saturation.py:device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tests/test_torch.py:device = "cuda:0" if torch.cuda.is_available() else "cpu"
delve/torchcallback.py:                               Default is cuda:0. Generally it is recommended
delve/torchcallback.py:                               on the gpu in order to get maximum performance.
delve/torchcallback.py:                               more limited VRAM of the GPU.
delve/torchcallback.py:                               GPUs, however delve itself will always
delve/torchcallback.py:                 device='cuda:0',
delve/torch_utils.py:                 device: str = 'cuda:0',
examples/example_lstm_generative_vae.py:    if torch.cuda.is_available():
examples/example_lstm_generative_vae.py:        x = x.cuda()
examples/example_lstm_generative_vae.py:                           self.hidden_dim).cuda(), torch.zeros(
examples/example_lstm_generative_vae.py:                               self.n_layers, bs, self.hidden_dim).cuda()
examples/example_lstm_generative_vae.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
examples/example_lstm_generative_vae.py:    if torch.cuda.is_available():
examples/example_lstm_generative_vae.py:        net.cuda()
examples/example_lstm_generative_vae.py:    #net = nn.DataParallel(net, device_ids=['cuda:0', 'cuda:1'])
examples/example_lstm_generative_vae.py:    eps = torch.Tensor([1e-10]).cuda()
examples/example_deep.py:from torch.cuda import is_available
examples/example_deep.py:    device = "cuda:0" if is_available() else "cpu"
examples/example_deep.py:            with torch.cuda.amp.autocast():
examples/example.py:device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
examples/example_fc.py:device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
examples/example_lstm_class_ae.py:    if torch.cuda.is_available():
examples/example_lstm_class_ae.py:        x = x.cuda()
examples/example_lstm_class_ae.py:                           self.hidden_dim).cuda(), torch.zeros(
examples/example_lstm_class_ae.py:                               self.n_layers, bs, self.hidden_dim).cuda()
examples/example_lstm_class_ae.py:    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
examples/example_lstm_class_ae.py:    if torch.cuda.is_available():
examples/example_lstm_class_ae.py:        net.cuda()
examples/example_lstm_class_ae.py:    #net = nn.DataParallel(net, device_ids=['cuda:0', 'cuda:1'])
examples/example_lstm_class_ae.py:    eps = torch.Tensor([1e-10]).cuda()

```
