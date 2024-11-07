# https://github.com/gperezs/StarcNet

```console
run_starcnet.sh:                   --cuda  --gpu 0 \
README.md:objects is about 4 mins on a CPU (4 secs with a GPU).
src/test_net.py:    parser.add_argument('--gpu', dest='gpu', help='CUDA visible device',
src/test_net.py:    parser.add_argument('--cuda', action='store_true', default=False,
src/test_net.py:                        help='enables CUDA training')
src/test_net.py:            if args.cuda:
src/test_net.py:                data, target = data.cuda(), target.cuda()
src/test_net.py:    args.cuda = args.cuda and torch.cuda.is_available()
src/test_net.py:    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
src/test_net.py:    if args.cuda:
src/test_net.py:        torch.cuda.manual_seed(args.seed)
src/test_net.py:        if args.cuda:
src/test_net.py:    if args.cuda:
src/test_net.py:        model.cuda()

```
