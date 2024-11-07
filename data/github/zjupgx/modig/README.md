# https://github.com/zjupgx/modig

```console
modig_graph.py:                         num_negative_samples=1, p=node2vec_p, q=node2vec_q, sparse=True).cuda()
modig_graph.py:                loss = model.loss(pos_rw.cuda(), neg_rw.cuda())
utils.py:    torch.cuda.manual_seed(seed)
utils.py:    torch.cuda.manual_seed_all(seed)
main.py:cuda = torch.cuda.is_available()
main.py:    graphlist_adj = [graph.cuda() for graph in graphlist]
main.py:            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())
main.py:            output[mask], label, pos_weight=torch.Tensor([2.7]).cuda())
main.py:                p.cuda() for p in k_sets[j][cv_run] if type(p) == torch.Tensor]
main.py:            model.cuda()
main.py:                torch.cuda.empty_cache()
main.py:    model.cuda()
main.py:        _, _ = train(all_mask.cuda(), all_label.cuda())
main.py:        torch.cuda.empty_cache()

```
