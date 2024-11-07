# https://github.com/Filter-Bubble/stroll

```console
stroll/stanza.py:    def __init__(self, config, pipeline, use_gpu):
utils/train_srl.py:# device = 'cuda' if torch.cuda.is_available() else 'cpu'
utils/run_stanza.py:        '--nogpu',
utils/run_stanza.py:        help='Disable GPU accelaration'
utils/run_stanza.py:                use_gpu=not args.nogpu
utils/run_stanza.py:                use_gpu=not args.nogpu
utils/run_stanza.py:                use_gpu=not args.nogpu
utils/pipeline.sh:python run_stanza.py --nogpu -f conllu --output ${doc}_stanza.conll ${filename}

```
