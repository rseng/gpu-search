# https://github.com/AprilYuge/ResPAN

```console
ResPAN/respan.py:    cuda = True if torch.cuda.is_available() else False 
ResPAN/respan.py:    if cuda: 
ResPAN/respan.py:        eta = eta.cuda() 
ResPAN/respan.py:    if cuda: 
ResPAN/respan.py:        interpolated = interpolated.cuda() 
ResPAN/respan.py:                                  prob_interpolated.size()).cuda() if cuda else torch.ones( 
ResPAN/respan.py:    if torch.cuda.is_available():
ResPAN/respan.py:        D = D.cuda()
ResPAN/respan.py:        G = G.cuda()
ResPAN/respan.py:        label_data_sample = torch.FloatTensor(label_data_sample).cuda()
ResPAN/respan.py:        train_data_sample = torch.FloatTensor(train_data_sample).cuda()
ResPAN/respan.py:    test_data = torch.FloatTensor(query_data).cuda()
ResPAN/respan.py:    torch.cuda.manual_seed_all(seed)

```
