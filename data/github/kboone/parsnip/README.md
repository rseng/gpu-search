# https://github.com/kboone/parsnip

```console
docs/photoz.rst:Note: Model training is much faster if a GPU is available. By default, ParSNIP will
docs/photoz.rst:attempt to use the GPU if there is one and fallback to CPU if not. This can be overriden
docs/boone2021.rst:Note: Model training is much faster if a GPU is available. By default, ParSNIP will
docs/boone2021.rst:attempt to use the GPU if there is one and fallback to CPU if not. This can be overriden
parsnip/utils.py:    elif device == 'cuda' and torch.cuda.is_available():
parsnip/utils.py:        use_device = 'cuda'
scripts/parsnip_train:    parser.add_argument('--device', default='cuda')
scripts/parsnip_predict:    parser.add_argument('--device', default='cuda')

```
