# https://github.com/biomedia-mira/blast-ct

```console
README.md:conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
README.md:indexing a cuda capable GPU on your machine. Defaults to CPU;
README.md: (recommended for gpu).
README.md:indexing a cuda capable GPU on your machine. Defaults to CPU;
README.md:Run the following in the `blast-ct-example` directory (GPU example):
README.md:    --device <gpu_id> \
README.md:6. `--device <device-id>` the device used for computation (`'cpu'` or integer indexing GPU). GPU is strongly recommended.
README.md:Run the following in the `blast-ct-example` directory (GPU example, takes time):
README.md:   --device <gpu_id> \
README.md:indexing a cuda capable GPU on your machine. Defaults to CPU;
README.md:Run the following in the `blast-ct-example` directory (GPU example):
blast_ct/read_config.py:def get_train_loader(config, model, train_csv_path, use_cuda):
blast_ct/read_config.py:                              pin_memory=True if use_cuda else False)
blast_ct/read_config.py:def get_valid_loader(config, model, test_csv_path, use_cuda):
blast_ct/read_config.py:                              pin_memory=True if use_cuda else False)
blast_ct/read_config.py:def get_test_loader(config, model, test_csv_path, use_cuda):
blast_ct/read_config.py:                             pin_memory=True if use_cuda else False)
blast_ct/inference.py:    use_cuda = device.type != 'cpu'
blast_ct/inference.py:    test_loader = get_test_loader(config, model, test_csv_path, use_cuda)
blast_ct/train.py:    use_cuda = device.type != 'cpu'
blast_ct/train.py:    train_loader = get_train_loader(config, model, train_csv_path, use_cuda)
blast_ct/train.py:    valid_loader = get_valid_loader(config, model, valid_csv_path, use_cuda)
blast_ct/train.py:    test_loader = get_test_loader(config, model, valid_csv_path, use_cuda)
blast_ct/train.py:    device = torch.device(device if torch.cuda.is_available() else 'cpu')
blast_ct/console_tool.py:    parser.add_argument('--device', help='GPU device index (int) or \'cpu\' (str)', default='cpu')
blast_ct/console_tool.py:    test_loader = get_test_loader(config, model, test_csv_path, use_cuda=not device.type == 'cpu')

```
