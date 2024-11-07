# https://github.com/ZhenjiangFan/DAG-deepVASE

```console
existingMethods/DAG-GNN/src/train.py:parser.add_argument('--no-cuda', action='store_true', default=True,
existingMethods/DAG-GNN/src/train.py:                    help='Disables CUDA training.')
existingMethods/DAG-GNN/src/train.py:args.cuda = not args.no_cuda and torch.cuda.is_available()
existingMethods/DAG-GNN/src/train.py:if args.cuda:
existingMethods/DAG-GNN/src/train.py:    torch.cuda.manual_seed(args.seed)
existingMethods/DAG-GNN/src/train.py:    if args.cuda:
existingMethods/DAG-GNN/src/train.py:        log_prior = log_prior.cuda()
existingMethods/DAG-GNN/src/train.py:if args.cuda:
existingMethods/DAG-GNN/src/train.py:    encoder.cuda()
existingMethods/DAG-GNN/src/train.py:    decoder.cuda()
existingMethods/DAG-GNN/src/train.py:    rel_rec = rel_rec.cuda()
existingMethods/DAG-GNN/src/train.py:    rel_send = rel_send.cuda()
existingMethods/DAG-GNN/src/train.py:    triu_indices = triu_indices.cuda()
existingMethods/DAG-GNN/src/train.py:    tril_indices = tril_indices.cuda()
existingMethods/DAG-GNN/src/train.py:        if args.cuda:
existingMethods/DAG-GNN/src/train.py:            data, relations = data.cuda(), relations.cuda()
existingMethods/DAG-GNN/src/utils.py:    if logits.is_cuda:
existingMethods/DAG-GNN/src/utils.py:        logistic_noise = logistic_noise.cuda()
existingMethods/DAG-GNN/src/utils.py:    if logits.is_cuda:
existingMethods/DAG-GNN/src/utils.py:        gumbel_noise = gumbel_noise.cuda()
existingMethods/DAG-GNN/src/utils.py:        if y_soft.is_cuda:
existingMethods/DAG-GNN/src/utils.py:            y_hard = y_hard.cuda()

```
