# https://github.com/yuhuan-wu/MobileSal

```console
parallel.py:import torch.cuda.comm as comm
parallel.py:    """Cross GPU all reduce autograd operation for calculate mean and
parallel.py:        ctx.target_gpus = [inputs[i].get_device() for i in range(0, len(inputs), num_inputs)]
parallel.py:        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
parallel.py:        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
parallel.py:        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
parallel.py:        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
parallel.py:        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
parallel.py:        return Broadcast.apply(ctx.target_gpus, gradOutput)
parallel.py:    The batch size should be larger than the number of GPUs used. It should
parallel.py:    also be an integer multiple of the number of GPUs so that each chunk is
parallel.py:    the same size (so that each GPU processes the same number of samples).
parallel.py:        device_ids: CUDA devices (default: all devices)
parallel.py:    Calculate loss in multiple-GPUs, which balance the memory usage for
parallel.py:            with torch.cuda.device(device):
SalEval.py:        self.thresh = torch.linspace(1./(nthresh + 1), 1. - 1./(nthresh + 1), nthresh).cuda()
SalEval.py:        self.gt_sum = torch.zeros((nthresh,)).cuda()
SalEval.py:        self.pred_sum = torch.zeros((nthresh,)).cuda()
SalEval.py:        self.prec = torch.zeros(self.nthresh).cuda()
SalEval.py:        self.recall = torch.zeros(self.nthresh).cuda()
SalEval.py:        recall = torch.zeros(self.nthresh).cuda()
SalEval.py:        prec = torch.zeros(self.nthresh).cuda()
README.md:* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
README.md:* [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt)
tools/test_trt.py:            depth = torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
tools/test_trt.py:        label = torch.from_numpy(label).float().unsqueeze(0).cuda()
tools/test_trt.py:        if args.gpu:
tools/test_trt.py:            img_variable = img_variable.cuda()
tools/test_trt.py:    if args.gpu:
tools/test_trt.py:                model = TRTModule().cuda()
tools/test_trt.py:                x = torch.randn(1,3,320,320).cuda()
tools/test_trt.py:                y = torch.randn(1,1,320,320).cuda()
tools/test_trt.py:                model = torch2trt(model.cuda(), [x,y])
tools/test_trt.py:    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
tools/test_trt.py:                        help='Run on CPU or GPU. If TRUE, then GPU')
tools/test.py:            depth = torch.from_numpy(depth).unsqueeze(dim=0).unsqueeze(dim=0).float().cuda()
tools/test.py:        label = torch.from_numpy(label).float().unsqueeze(0).cuda()
tools/test.py:        if args.gpu:
tools/test.py:            img_variable = img_variable.cuda()
tools/test.py:    if args.gpu:
tools/test.py:        model = model.cuda()
tools/test.py:    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
tools/test.py:                        help='Run on CPU or GPU. If TRUE, then GPU')
tools/train.sh:CUDA_VISIBLE_DEVICES=0 python3 tools/train.py --file_root ./data/ \
tools/train.py:        if args.onGPU:
tools/train.py:            input = input.cuda()
tools/train.py:            target = target.cuda()
tools/train.py:                depth = depth.cuda()
tools/train.py:        #torch.cuda.synchronize()
tools/train.py:        if args.onGPU and torch.cuda.device_count() > 1:
tools/train.py:        if args.onGPU == True:
tools/train.py:            input = input.cuda()
tools/train.py:            target = target.cuda()
tools/train.py:                depth = depth.cuda()
tools/train.py:        if args.onGPU and torch.cuda.device_count() > 1:
tools/train.py:        # Computing F-measure and MAE on GPU
tools/train.py:    if args.onGPU and torch.cuda.device_count() > 1:
tools/train.py:    if args.onGPU:
tools/train.py:        model = model.cuda()
tools/train.py:    if args.onGPU:
tools/train.py:    if args.onGPU and torch.cuda.device_count() > 1:
tools/train.py:            torch.cuda.empty_cache()
tools/train.py:            torch.cuda.empty_cache()
tools/train.py:        torch.cuda.empty_cache()
tools/train.py:        torch.cuda.empty_cache()
tools/train.py:        torch.cuda.empty_cache()
tools/train.py:    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
tools/train.py:                        help='Run on CPU or GPU. If TRUE, then GPU.')
speed_test.py:model = MobileSal().cuda().eval()
speed_test.py:x = torch.randn(20,3,320,320).cuda()
speed_test.py:y = torch.randn(20,1,320,320).cuda()
speed_test.py:    p = p + 1 # replace torch.cuda.synchronize()
speed_test.py:torch.cuda.empty_cache()
speed_test.py:x = torch.randn(1,3,320,320).cuda()
speed_test.py:y = torch.randn(1,1,320,320).cuda()
speed_test.py:torch.cuda.empty_cache()
speed_test.py:        p = p + 1 # replace torch.cuda.synchronize()
speed_test.py:    p = p + 1 # replace torch.cuda.synchronize()
criteria.py:    return window.cuda()

```
