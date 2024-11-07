# https://github.com/DeepRank/Deeprank-GNN

```console
docs/conf.py:    'torch.cuda',
README.md:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deeprank_gnn/NeuralNet.py:            'cuda' if torch.cuda.is_available() else 'cpu')
deeprank_gnn/NeuralNet.py:        if self.device.type == 'cuda':
deeprank_gnn/NeuralNet.py:            print(torch.cuda.get_device_name(0))
deeprank_gnn/NeuralNet.py:            'cuda' if torch.cuda.is_available() else 'cpu')
example/model.py:device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

```
