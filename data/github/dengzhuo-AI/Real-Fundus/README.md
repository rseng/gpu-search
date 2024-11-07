# https://github.com/dengzhuo-AI/Real-Fundus

```console
train_code/architecture/RFormer.py:#    with torch.cuda.device(0):
train_code/architecture/RFormer.py:#summary(RFormer_D().cuda(),torch.zeros((1,3,128,128)).cuda())
train_code/train.yml:GPU: [0]
train_code/train.py:gpus = ','.join([str(i) for i in opt.GPU])
train_code/train.py:os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
train_code/train.py:os.environ["CUDA_VISIBLE_DEVICES"] = gpus
train_code/train.py:torch.cuda.manual_seed_all(1234)
train_code/train.py:model_g.cuda()
train_code/train.py:model_d.cuda()
train_code/train.py:device_ids = [i for i in range(torch.cuda.device_count())]
train_code/train.py:if torch.cuda.device_count() > 1:
train_code/train.py:  logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))
train_code/train.py:  logger.info('GPUs:{}'.format(gpus))
train_code/train.py:lossc = CharbonnierLoss().cuda()
train_code/train.py:lossp = PerceptualLoss().cuda()
train_code/train.py:losse = EdgeLoss().cuda()
train_code/train.py:real_labels_patch = Variable(torch.ones(opt.OPTIM.BATCH_SIZE, 169) - 0.05).cuda()
train_code/train.py:fake_labels_patch = Variable(torch.zeros(opt.OPTIM.BATCH_SIZE, 169)).cuda()
train_code/train.py:        target = data[0].cuda()
train_code/train.py:        input_ = data[1].cuda()
train_code/train.py:                    target = data_val[0].cuda()
train_code/train.py:                    input_ = data_val[1].cuda()
train_code/loss.py:        self.weights = torch.tensor(weights).cuda()
train_code/loss.py:        vgg = vgg.cuda()
train_code/loss.py:            lhs = lhs.cuda()
train_code/loss.py:            rhs = rhs.cuda()
train_code/loss.py:        if torch.cuda.is_available():
train_code/loss.py:            self.kernel = self.kernel.cuda()
train_code/utils.py:        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()
train_code/utils.py:def load_checkpoint_multigpu(model, weights):
README.md:|[GLCAE](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w43/Tian_Global_and_Local_ICCV_2017_paper.pdf) | --- | --- | 21.37 | 0.570 | |
README.md:| [ESRGAN](https://openaccess.thecvf.com/content_eccv_2018_workshops/w25/html/Wang_ESRGAN_Enhanced_Super-Resolution_Generative_Adversarial_Networks_ECCVW_2018_paper.html) | 15.95 | 18.41 | 26.73 | 0.823 |   |
README.md:| [RealSR](https://openaccess.thecvf.com/content_CVPRW_2020/html/w31/Ji_Real-World_Super-Resolution_via_Kernel_Estimation_and_Noise_Injection_CVPRW_2020_paper.html) | 15.92 | 29.42 | 27.99 | 0.850 |   |
README.md:| [MST](https://openaccess.thecvf.com/content/CVPR2022/html/Cai_Mask-Guided_Spectral-Wise_Transformer_for_Efficient_Hyperspectral_Image_Reconstruction_CVPR_2022_paper.html) | 3.48 | 3.59| 28.13 | 0.854 |    |
README.md:+ NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
test_code/architecture/RFormer.py:#    with torch.cuda.device(0):
test_code/architecture/RFormer.py:#summary(RFormer_D().cuda(),torch.zeros((1,3,128,128)).cuda())
test_code/test.py:os.environ["CUDA_VISIBLE_DEVICES"] = "0"
test_code/test.py:device_ids = [i for i in range(torch.cuda.device_count())]
test_code/test.py:model_restoration.cuda()
test_code/test.py:criterion = pytorch_ssim.SSIM(window_size = 3).cuda()
test_code/test.py:        rgb_gt = data_test[0].cuda()
test_code/test.py:        rgb_noisy = data_test[1].cuda()
test_code/utils.py:        lam = self.dist.rsample((bs,1)).view(-1,1,1,1).cuda()
test_code/utils.py:def load_checkpoint_multigpu(model, weights):

```
