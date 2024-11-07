# https://github.com/fhaghighi/TransVW

```console
keras/train.py:    # To find a largest batch size that can be fit into GPU
README.md:This research has been supported partially by ASU and Mayo Clinic through a Seed Grant and an Innovation Grant, and partially by the NIH under Award Number R01HL128785. The content is solely the responsibility of the authors and does not necessarily represent the official views of the NIH. This work has utilized the GPUs provided partially by the ASU Research Computing and partially by the Extreme Science and Engineering Discovery Environment (XSEDE) funded by the National Science Foundation (NSF) under grant number ACI-1548562. We thank [Zuwei Guo](https://www.linkedin.com/in/zuwei/) for implementing Rubik's cube, [M. M. Rahman Siddiquee](https://github.com/mahfuzmohammad) for examining NiftyNet, and [Jiaxuan Pang](https://www.linkedin.com/in/jiaxuan-pang-b014ab127/) for evaluating I3D, [Shivam Bajpai](https://github.com/sbajpai2) for helping in adopting TransVW to nnU-Net, and
requirements.txt:tensorflow-gpu>=1.13.1
pytorch/README.md:target_model = nn.DataParallel(target_model, device_ids = [i for i in range(torch.cuda.device_count())])
pytorch/README.md:model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
pytorch/train.py:parser.add_argument('--multi_gpu', action='store_true', default=False, help='use multi gpu?')
pytorch/train.py:device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pytorch/train.py:print(torch.cuda.device_count())
pytorch/train.py:if args.multi_gpu:
pytorch/train.py:    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
pytorch/train.py:print("Total CUDA devices: ", torch.cuda.device_count() ,file=conf.log_writter)
pytorch/train.py:		    torch.cuda.empty_cache()
pytorch/train.py:		torch.cuda.empty_cache()

```
