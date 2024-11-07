# https://github.com/biomedia-mira/istn

```console
istn-reg.py:    use_cuda = torch.cuda.is_available()
istn-reg.py:    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
istn-reg.py:    if use_cuda:
istn-reg.py:        print('GPU: ' + str(torch.cuda.get_device_name(int(args.dev))))
istn-reg.py:    parser.add_argument('--dev', default='0', help='cuda device (default: 0)')
pymira/nets/stn.py:        # Cuda params
pymira/nets/stn.py:        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
pymira/nets/stn.py:        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float
pymira/nets/stn.py:        # Cuda params
pymira/nets/stn.py:        self.dtype = torch.cuda.float if (self.device == 'cuda') else torch.float

```
