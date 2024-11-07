# https://github.com/biomed-AI/GraphBepi

```console
graph_construction.py:    # C=coord.to('cuda:1')
test.py:    torch.cuda.manual_seed(seed)
test.py:parser.add_argument('--gpu', type=int, default=0, help='gpu.')
test.py:device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
test.py:            # if e.args[0].startswith("CUDA out of memory"):
test.py:            #     print(f"Failed (CUDA out of memory) on sequence {i} of length {len(j)}.")
test.py:trainer = pl.Trainer(gpus=[args.gpu],logger=None)
README.md:2. `python dataset.py --gpu 0`
README.md:It will take about 20 minutes to download the pretrained ESM-2 model and an hour to build our dataset with CUDA.
README.md:python test.py -i pdb_file -p --gpu 0 -o ./output
README.md:python test.py -i fasta_file -f --gpu 0 -o ./output
train.py:    torch.cuda.manual_seed(seed)
train.py:parser.add_argument('--gpu', type=int, default=0, help='gpu.')
train.py:device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'
train.py:    gpus=[args.gpu] if args.gpu!=-1 else None, 
train.py:trainer = pl.Trainer(gpus=[args.gpu],logger=None)
dataset.py:    parser.add_argument('--gpu', type=int, default=0, help='gpu.')
dataset.py:    device='cpu' if args.gpu==-1 else f'cuda:{args.gpu}'

```
