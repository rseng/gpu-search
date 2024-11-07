# https://github.com/JieZheng-ShanghaiTech/PiLSL

```console
README.md:    --gpu=0              # ID of GPU
train.py:os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
train.py:    parser.add_argument("--gpu", type=int, default=2,
train.py:                        help="Which GPU to use?")
train.py:    parser.add_argument('--disable_cuda', action='store_true',
train.py:                        help='Disable CUDA')
train.py:    if not params.disable_cuda and torch.cuda.is_available():
train.py:        params.device = torch.device('cuda:%d' % params.gpu)

```
