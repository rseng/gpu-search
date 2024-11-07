# https://github.com/ericcombiolab/LRTK

```console
script/Pangaea/models/VAENET.py:    def __init__(self, abd_dim, tnf_dim, latent_size, num_classes, epochs, cuda, num_gpus, lr, dropout, alpha, w_kl, weight_decay):
script/Pangaea/models/VAENET.py:        self.cuda = cuda
script/Pangaea/models/VAENET.py:        if self.cuda:
script/Pangaea/models/VAENET.py:            self.network = self.network.cuda()
script/Pangaea/models/VAENET.py:            self.network = nn.DataParallel(self.network, device_ids=range(num_gpus))
script/Pangaea/models/VAENET.py:                    if self.cuda:
script/Pangaea/models/VAENET.py:                        abd = abd.cuda()
script/Pangaea/models/VAENET.py:                        tnf = tnf.cuda()
script/Pangaea/models/VAENET.py:                                if self.cuda:
script/Pangaea/models/VAENET.py:                                    abd = abd.cuda()
script/Pangaea/models/VAENET.py:                                    tnf = tnf.cuda()
script/Pangaea/models/VAENET.py:                        if self.cuda:
script/Pangaea/models/VAENET.py:                            abd = abd.cuda()
script/Pangaea/models/VAENET.py:                            tnf = tnf.cuda()
script/Pangaea/models/VAENET.py:                    if self.cuda:
script/Pangaea/models/VAENET.py:                        abd = abd.cuda()
script/Pangaea/models/VAENET.py:                        tnf = tnf.cuda()
script/Pangaea/athena_environment.yaml:  - magpurify=1.0=2
script/Pangaea/pangaea.py:                epochs=args.epochs, cuda=args.use_cuda, num_gpus=args.num_gpus, lr=args.lr, dropout=args.dropout,
script/Pangaea/pangaea.py:    parser.add_argument("-g", "--use_cuda", type=bool, default=False, help="use cuda (default False)")
script/Pangaea/pangaea.py:    parser.add_argument("-n", "--num_gpus", type=int, default=1, help="use gpu in parallel (if use cuda)")
script/Pangaea/README.md:                  [-ld LATENT_DIM] -c CLUSTERS [-t THREADS] [-g USE_CUDA]
script/Pangaea/README.md:                  [-n NUM_GPUS] [-sp SPADES] [-lc LOCAL_ASSEMBLY] [-at ATHENA]
script/Pangaea/README.md:  -g USE_CUDA, --use_cuda USE_CUDA
script/Pangaea/README.md:                        use cuda (default False)
script/Pangaea/README.md:  -n NUM_GPUS, --num_gpus NUM_GPUS
script/Pangaea/README.md:                        use gpu in parallel (if use cuda)
script/Pangaea/environment.yaml:  - cudatoolkit=11.3.1=h9edb442_11
script/Pangaea/environment.yaml:  - pytorch=1.10.0=py3.8_cuda11.3_cudnn8.2.0_0
script/Pangaea/environment.yaml:  - pytorch-mutex=1.0=cuda
script/Pangaea/utils.py:	torch.cuda.manual_seed_all(seed)

```
