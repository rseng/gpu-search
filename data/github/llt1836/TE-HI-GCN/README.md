# https://github.com/llt1836/TE-HI-GCN

```console
train-ehigcn.py:def train(name, brain_region_num, device='cuda'):
train-ehigcn.py:    if torch.cuda.is_available():
train-ehigcn.py:        device = 'cuda'
train-higcn.py:def train(name, device='cuda'):
train-higcn.py:    if torch.cuda.is_available():
train-higcn.py:        device = 'cuda'
adni_new/train_adni_fgcn.py:        eye = torch.eye(feature_dim).cuda()
adni_new/train_adni_fgcn.py:    if torch.cuda.is_available():
adni_new/train_adni_fgcn.py:        device = 'cuda'
adni_new/train_adni_higcn.py:    if torch.cuda.is_available():
adni_new/train_adni_higcn.py:        device = 'cuda'
adni_new/train_adni_ehigcn.py:    if torch.cuda.is_available():
adni_new/train_adni_ehigcn.py:        device = 'cuda'
adni_new/hinets/model_higcn_ADNI.py:        eye = torch.eye(feature_dim).cuda()
adni_new/hinets/model_higcn_ADNI.py:    def __init__(self, in_dim, hidden_dim, graph_adj, num, thr=0, num_feats=50, num_nodes=871, num_classes=2, device='cuda'):
adni_new/hinets/model_higcn_ADNI.py:    def forward(self, nodes, nodes_adj, device='cuda'):
adni_new/hinets/model_ehigcn_adni.py:        eye = torch.eye(feature_dim).cuda()
adni_new/hinets/model_ehigcn_adni.py:    def __init__(self, in_dim, hidden_dim, graph_adj, num, thr='0', num_feats=50, num_classes=2, device='cuda'):
adni_new/hinets/model_ehigcn_adni.py:    def forward(self, nodes, nodes_adj, device='cuda'):
adni_new/hinets/model_ehigcn_adni.py:                 num_feats=50, num_classes=2, device='cuda'):
adni_new/hinets/model_ehigcn_adni.py:    def forward(self, nodes, nodes_adj, device='cuda'):
train-fgcn.py:        eye = torch.eye(feature_dim).cuda()
train-fgcn.py:    print(torch.cuda.is_available())
train-fgcn.py:    if torch.cuda.is_available():
train-fgcn.py:        device = 'cuda:0'
train-fgcn.py:                                                        test_dataset=test_data_loaders[i], device='cuda')
train-fgcn.py:        _, test_result = evaluate(test_data_loaders[i], model, name='Test', device='cuda')
hinets/model_higcn.py:        eye = torch.eye(feature_dim).cuda()
hinets/model_higcn.py:                 brain_region_num=116, name='aal', device='cuda'):
hinets/model_higcn.py:    def forward(self, nodes, nodes_adj, device='cuda'):
hinets/model_ehigcn.py:        eye = torch.eye(feature_dim).cuda()
hinets/model_ehigcn.py:                 brain_region_num=116, name='aal', device='cuda'):
hinets/model_ehigcn.py:    def forward(self, nodes, nodes_adj, device='cuda'):
hinets/model_ehigcn.py:                 num_feats=50, num_classes=2, brain_region_num=116, name='aal', device='cuda'):
hinets/model_ehigcn.py:    def forward(self, nodes, nodes_adj, device='cuda'):
hinets/model_fgcn.py:        eye = torch.eye(feature_dim).cuda()

```
