# https://github.com/liulizhi1996/HPODNets

```console
README.md:Our model is implemented by Python 3.6 with Pytorch 1.4.0 and Pytorch-geometric 1.5.0, and run on Nvidia GPU with CUDA 10.0.
README.md:- `main.py`: Run the `main.py` script, and you will obtain the prediction results. Be careful to the cuda device ID!
src/method/main.py:t.cuda.set_device(0)
src/method/main.py:                "train_target": t.FloatTensor(train_target).cuda(),
src/method/main.py:                "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
src/method/main.py:                "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
src/method/main.py:            model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()
src/method/main.py:            "train_target": t.FloatTensor(train_target).cuda(),
src/method/main.py:            "feature": t.stack([t.FloatTensor(feat) for feat in protein_features]).cuda(),
src/method/main.py:            "network": t.stack([t.FloatTensor(net) for net in networks]).cuda(),
src/method/main.py:        model = Model(train_annotation.shape[0], train_annotation.shape[1], len(networks)).cuda()

```
