# https://github.com/tkzeng/Pangolin

```console
pangolin/pangolin.py:    if torch.cuda.is_available():
pangolin/pangolin.py:        ref_seq = ref_seq.to(torch.device("cuda"))
pangolin/pangolin.py:        alt_seq = alt_seq.to(torch.device("cuda"))
pangolin/pangolin.py:    if torch.cuda.is_available():
pangolin/pangolin.py:        print("Using GPU")
pangolin/pangolin.py:            if torch.cuda.is_available():
pangolin/pangolin.py:                model.cuda()
pangolin/.fuse_hidden0000252700000002:    if torch.cuda.is_available():
pangolin/.fuse_hidden0000252700000002:        ref_seq = ref_seq.to(torch.device("cuda"))
pangolin/.fuse_hidden0000252700000002:        alt_seq = alt_seq.to(torch.device("cuda"))
pangolin/.fuse_hidden0000252700000002:    if torch.cuda.is_available():
pangolin/.fuse_hidden0000252700000002:        print("Using GPU")
pangolin/.fuse_hidden0000252700000002:            if torch.cuda.is_available():
pangolin/.fuse_hidden0000252700000002:                model.cuda()
README.md:Pangolin can be run on Google Colab, which provides free acess to GPUs and other computing resources: https://colab.research.google.com/github/tkzeng/Pangolin/blob/main/PangolinColab.ipynb
README.md:  * If a supported GPU is available, installation with GPU support is recommended (choose an option under "Compute Platform")
scripts/custom_usage.py:        if torch.cuda.is_available():
scripts/custom_usage.py:            model.cuda()
scripts/custom_usage.py:    if torch.cuda.is_available():
scripts/custom_usage.py:        seq = seq.to(torch.device("cuda"))

```
