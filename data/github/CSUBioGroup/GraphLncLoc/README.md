# https://github.com/CSUBioGroup/GraphLncLoc

```console
models/classifier.py:    t.cuda.manual_seed_all(seed)
predict.py:mapLocation={'cuda:0':'cpu', 'cuda:1':'cpu'}
README.md:>>***device*** is the device you used to build and train the model. It can be "cpu" for cpu or "cuda" for gpu, and "cuda:0" for gpu 0.  
README.md:mapLocation={'cuda:0':'cpu', 'cuda:1':'cpu'}
variantModel/predict_dc.py:mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
variantModel/predict_5.py:mapLocation = {'cuda:0': 'cpu', 'cuda:1': 'cpu'}
variantModel/README.md:mapLocation={'cuda:0':'cpu', 'cuda:1':'cpu'}
utils/FocalLoss.py:#                             [2.5, 1]], device='cuda')
utils/FocalLoss.py:#     targets = torch.tensor([0, 1], device='cuda')
utils/config.py:        self.device = t.device("cuda:0")

```
