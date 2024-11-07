# https://github.com/SJLeo/FFSD

```console
ImageNetTrainer.py:        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'
CIFARTrainer.py:        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'
README.md:The code has been tested using Pytorch1.5.1 and CUDA10.2 on Ubuntu 18.04.
README.md:  	--gpu_ids 0
README.md:  	--gpu_ids 0,1
README.md:	--gpu_ids 0
DistributeImageNetTrainer.py:        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'
DistributeImageNetTrainer.py:            model = nn.DataParallel(model, device_ids=self.opt.gpu_ids)
DistributeImageNetTrainer.py:        self.leader_model = nn.DataParallel(self.leader_model, device_ids=self.opt.gpu_ids)
DistributeImageNetTrainer.py:            sd_model = nn.DataParallel(sd_model, device_ids=self.opt.gpu_ids)
DistributeImageNetTrainer.py:        self.sd_leader_model = nn.DataParallel(self.sd_leader_model, device_ids=self.opt.gpu_ids)
DistributeImageNetTrainer.py:        self.fusion_module = nn.DataParallel(self.fusion_module, device_ids=self.opt.gpu_ids)
DistributeImageNetTrainer.py:            for i in self.opt.gpu_ids:
DistributeImageNetTrainer.py:                name = layer + 'cuda:%d' % i
options/options.py:parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
options/options.py:    str_ids = opt.gpu_ids.split(',')
options/options.py:    opt.gpu_ids = []
options/options.py:            opt.gpu_ids.append(id)
Tester.py:        self.device = torch.device(f'cuda:{opt.gpu_ids[0]}') if torch.cuda.is_available() else 'cpu'

```
