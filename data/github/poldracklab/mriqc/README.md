# https://github.com/nipreps/mriqc

```console
mriqc/synthstrip/cli.py:    parser.add_argument('-g', '--gpu', action='store_true', help='Use the GPU.')
mriqc/synthstrip/cli.py:    # configure GPU device
mriqc/synthstrip/cli.py:    if args.gpu:
mriqc/synthstrip/cli.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
mriqc/synthstrip/cli.py:        device = torch.device('cuda')
mriqc/synthstrip/cli.py:        device_name = 'GPU'
mriqc/synthstrip/cli.py:        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
mriqc/interfaces/synthstrip.py:    use_gpu = traits.Bool(False, usedefault=True, argstr='-g', desc='Use GPU', nohash=True)
docker/files/neurodebian.gpg:I+O/lRsm6L9lc6rV0IgPU00P4BAwR+x8Rw7TJFbuS0miR3lP1NSguz+/kpjxzmGP

```
