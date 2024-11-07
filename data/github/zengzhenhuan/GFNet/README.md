# https://github.com/zengzhenhuan/GFNet

```console
Code/model_lung_infection/VGG.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/VGG.py:    input_tensor = torch.randn(1,3, 352, 352).cuda()
Code/model_lung_infection/backbone/Res2Net.py:    images = torch.rand(1, 3, 224, 224).cuda(0)
Code/model_lung_infection/backbone/Res2Net.py:    model = model.cuda(0)
Code/model_lung_infection/module/networks_other.py:             gpu_ids=[]):
Code/model_lung_infection/module/networks_other.py:    use_gpu = len(gpu_ids) > 0
Code/model_lung_infection/module/networks_other.py:    if use_gpu:
Code/model_lung_infection/module/networks_other.py:        assert (torch.cuda.is_available())
Code/model_lung_infection/module/networks_other.py:                               gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:                               gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:                             gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:                             gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:    if len(gpu_ids) > 0:
Code/model_lung_infection/module/networks_other.py:        netG.cuda(gpu_ids[0])
Code/model_lung_infection/module/networks_other.py:             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', gpu_ids=[]):
Code/model_lung_infection/module/networks_other.py:    use_gpu = len(gpu_ids) > 0
Code/model_lung_infection/module/networks_other.py:    if use_gpu:
Code/model_lung_infection/module/networks_other.py:        assert (torch.cuda.is_available())
Code/model_lung_infection/module/networks_other.py:                                   gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:                                   gpu_ids=gpu_ids)
Code/model_lung_infection/module/networks_other.py:    if use_gpu:
Code/model_lung_infection/module/networks_other.py:        netD.cuda(gpu_ids[0])
Code/model_lung_infection/module/networks_other.py:    # synchronize gpu time and measure fp
Code/model_lung_infection/module/networks_other.py:    torch.cuda.synchronize()
Code/model_lung_infection/module/networks_other.py:    torch.cuda.synchronize()
Code/model_lung_infection/module/networks_other.py:    torch.cuda.synchronize()
Code/model_lung_infection/module/networks_other.py:    # transfer the model_lung_infection on GPU
Code/model_lung_infection/module/networks_other.py:    model.cuda()
Code/model_lung_infection/module/networks_other.py:                 gpu_ids=[], padding_type='reflect'):
Code/model_lung_infection/module/networks_other.py:        self.gpu_ids = gpu_ids
Code/model_lung_infection/module/networks_other.py:        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
Code/model_lung_infection/module/networks_other.py:            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
Code/model_lung_infection/module/networks_other.py:                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
Code/model_lung_infection/module/networks_other.py:        self.gpu_ids = gpu_ids
Code/model_lung_infection/module/networks_other.py:        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
Code/model_lung_infection/module/networks_other.py:            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
Code/model_lung_infection/module/networks_other.py:    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
Code/model_lung_infection/module/networks_other.py:        self.gpu_ids = gpu_ids
Code/model_lung_infection/module/networks_other.py:        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
Code/model_lung_infection/module/networks_other.py:            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
Code/model_lung_infection/InfNet_VGGNet.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/InfNet_VGGNet.py:    input_tensor = torch.randn(1, 3, 352, 352).cuda()
Code/model_lung_infection/InfNet_ResNet.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/InfNet_ResNet.py:    input_tensor = torch.randn(1, 3, 352, 352).cuda()
Code/model_lung_infection/mynet_Res2Net.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/mynet_Res2Net.py:    input_tensor = torch.randn(1, 3, 352, 352).cuda()
Code/model_lung_infection/mynet_VGG.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/mynet_VGG.py:    input_tensor = torch.randn(1,3, 352, 352).cuda()
Code/model_lung_infection/InfNet_Res2Net.py:    ras = PraNetPlusPlus().cuda()
Code/model_lung_infection/InfNet_Res2Net.py:    input_tensor = torch.randn(1, 3, 352, 352).cuda()
Test_UNet.py:    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
Test_UNet.py:    model.cuda()
Test_UNet.py:        image = image.cuda()
Test_UNet++.py:    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
Test_UNet++.py:    model.cuda()
Test_UNet++.py:        image = image.cuda()
Test_LungInf.py:    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
Test_LungInf.py:    model.cuda()
Test_LungInf.py:        image = image.cuda()
mytest_VGG.py:    # model = torch.nn.DataParallel(model, device_ids=[0, 1]) # uncomment it if you have multiply GPUs.
mytest_VGG.py:    model.cuda()
mytest_VGG.py:        image = image.cuda()
Train_Unet++.py:            images = Variable(images).cuda()
Train_Unet++.py:            gts = Variable(gts).cuda()
Train_Unet++.py:            edges = Variable(edges).cuda()
Train_Unet++.py:    parser.add_argument('--gpu_device', type=int, default=0,
Train_Unet++.py:                        help='choose which GPU device you want to use')
Train_Unet++.py:    torch.cuda.set_device(opt.gpu_device)
Train_Unet++.py:    model = UNet_2Plus(in_channels=opt.net_channel, n_classes=opt.n_classes, feature_scale=4, is_deconv=True, is_batchnorm=True, is_ds=True).cuda()
Train_Unet++.py:        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
Train_Unet.py:            images = Variable(images).cuda()
Train_Unet.py:            gts = Variable(gts).cuda()
Train_Unet.py:            edges = Variable(edges).cuda()
Train_Unet.py:    parser.add_argument('--gpu_device', type=int, default=0,
Train_Unet.py:                        help='choose which GPU device you want to use')
Train_Unet.py:    torch.cuda.set_device(opt.gpu_device)
Train_Unet.py:    model = Inf_Net_UNet(n_channels=opt.net_channel, n_classes=opt.n_classes).cuda()
Train_Unet.py:        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
mytrain_VGG.py:            images = Variable(images).cuda()
mytrain_VGG.py:            gts = Variable(gts).cuda()
mytrain_VGG.py:            edges = Variable(edges).cuda()
mytrain_VGG.py:    parser.add_argument('--gpu_device', type=int, default=0,
mytrain_VGG.py:                        help='choose which GPU device you want to use')
mytrain_VGG.py:    torch.cuda.set_device(opt.gpu_device)
mytrain_VGG.py:    model = mynetVGG(channel=opt.net_channel, n_class=opt.n_classes).cuda()
mytrain_VGG.py:        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()
Train_LungInf_Res2Net.py:            images = Variable(images).cuda()
Train_LungInf_Res2Net.py:            gts = Variable(gts).cuda()
Train_LungInf_Res2Net.py:            edges = Variable(edges).cuda()
Train_LungInf_Res2Net.py:    parser.add_argument('--gpu_device', type=int, default=0,
Train_LungInf_Res2Net.py:                        help='choose which GPU device you want to use')
Train_LungInf_Res2Net.py:    torch.cuda.set_device(opt.gpu_device)
Train_LungInf_Res2Net.py:    model = Inf_Net(channel=opt.net_channel, n_class=opt.n_classes).cuda()
Train_LungInf_Res2Net.py:        x = torch.randn(1, 3, opt.trainsize, opt.trainsize).cuda()

```
