# https://github.com/PRIS-CV/DS-UI

```console
README.md:- GPU memory >= 5000MiB (GTX 1080Ti)
README.md:- card: Index of the used GPU
main.py:os.environ["CUDA_VISIBLE_DEVICES"] = args.card
main.py:use_cuda = torch.cuda.is_available()
main.py:device = torch.device("cuda")
gmm_layer.py:            self.pi = torch.ones([1, 1, self.num_out]).cuda()# unormalized weights
gmm_layer.py:        self.omega = torch.zeros([1, self.num_out]).cuda()
gmm_layer.py:                                                prior_sigma.cuda()))
gmm_layer.py:        one_hot_label = torch.zeros_like(outputs).cuda() - 1.# [b x c]
gmm_layer.py:        one_hot_label[torch.arange(b), targets] = torch.ones([b]).cuda()
gmm_layer.py:        one_hot_label = one_hot_label.cuda()

```
