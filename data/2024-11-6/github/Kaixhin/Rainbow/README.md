# https://github.com/Kaixhin/Rainbow

```console
agent.py:        state_dict = torch.load(args.model, map_location='cpu')  # Always load tensors onto CPU by default, will shift to GPU if necessary
main.py:parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
main.py:if torch.cuda.is_available() and not args.disable_cuda:
main.py:  args.device = torch.device('cuda')
main.py:  torch.cuda.manual_seed(np.random.randint(1, 10000))

```
