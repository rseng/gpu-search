# https://github.com/ruizengalways/FOD-Net

```console
config/fodnet_updated_config.yml:# Learning rate and GPU IDs
config/fodnet_updated_config.yml:gpu_ids: 3
models/base_model.py:        self.gpu_ids = opt.gpu_ids
models/base_model.py:        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
models/base_model.py:                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
models/base_model.py:                    net.cuda(self.gpu_ids[0])
models/networks.py:def define_network(init_type='normal', init_gain=1., gpu_ids=[]):
models/networks.py:    return init_net(net, init_type, init_gain, gpu_ids)
models/networks.py:def init_net(net, init_type='kaiming', init_gain=0.02, gpu_ids=[]):
models/networks.py:    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
models/networks.py:        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
models/networks.py:    if len(gpu_ids) > 0:
models/networks.py:        assert (torch.cuda.is_available())
models/networks.py:        net.to(gpu_ids[0])
models/networks.py:        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
models/fodnet_model.py:                                              gpu_ids=self.gpu_ids)
test.py:    gpu_ids = opt.gpu_ids
README.md:- CPU or NVIDIA GPU + CUDA CuDNN
README.md:python test.py --fod_path ./dataset/${subject_id}/ss3t_csd/WM_FODs_normalised.nii.gz --weights_path ./fodnet.pth --output_path ./dataset/${subject_id}/fodnet/SR_WM_FODs_normalised.nii.gz --brain_mask_path ./dataset/${subject_id}/brain_mask.nii.gz --gpu_ids 0
options/base_options.py:        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
options/base_options.py:        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
options/base_options.py:        # set gpu ids
options/base_options.py:        str_ids = opt.gpu_ids.split(',')
options/base_options.py:        opt.gpu_ids = []
options/base_options.py:                opt.gpu_ids.append(id)
options/base_options.py:        if len(opt.gpu_ids) > 0:
options/base_options.py:            torch.cuda.set_device(opt.gpu_ids[0])

```
