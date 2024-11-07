# https://github.com/Kaixhin/PlaNet

```console
main.py:parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
main.py:if torch.cuda.is_available() and not args.disable_cuda:
main.py:  args.device = torch.device('cuda')
main.py:  torch.cuda.manual_seed(args.seed)

```
