# https://github.com/bbbbbbzhou/MDPET

```console
networks/layers.py:        # registering the grid as a buffer cleanly moves it to the GPU, but it also
networks/convolutional_rnn/functional.py:    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
networks/convolutional_rnn/functional.py:    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
networks/__init__.py:def set_gpu(network, gpu_ids):
networks/__init__.py:    network.to(gpu_ids[0])
networks/__init__.py:    network = nn.DataParallel(network, device_ids=gpu_ids)
networks/__init__.py:    return set_gpu(network, opts.gpu_ids)
networks/__init__.py:    return set_gpu(network, opts.gpu_ids)
models/model_svrhd_dp_gan.py:            self.loss_dn_gan = LSGANLoss().cuda()
models/model_svrhd_dp_gan.py:    def setgpu(self, gpu_ids):
models/model_svrhd_dp_gan.py:        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))
models/utils.py:                                      rotation_resolution=fan_rotation_inremental).cuda()
models/losses.py:        sum_filt = torch.ones([1, 1, *win]).to("cuda")
test.py:parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
test.py:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test.py:model.setgpu(opts.gpu_ids)
README.md:Our code has been tested with Python 3.7, Pytorch 1.4.0, CUDA 10.0 on Ubuntu 18.04.
scripts/train_svrhddn_dp_gan_mse.sh:CUDA_VISIBLE_DEVICES=1 python train.py \
train.py:parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0], help='list of gpu ids')
train.py:device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train.py:model.setgpu(opts.gpu_ids)
utils/+helper/train_image2image_dag.m:opts.gpus = [] ;
utils/+helper/train_image2image_dag.m:  prepareGPUs(opts, epoch == start+1) ;
utils/+helper/train_image2image_dag.m:  if numel(opts.gpus) <= 1
utils/+helper/train_image2image_dag.m:% With multiple GPUs, return one copy
utils/+helper/train_image2image_dag.m:% move CNN  to GPU as needed
utils/+helper/train_image2image_dag.m:numGpus = numel(params.gpus) ;
utils/+helper/train_image2image_dag.m:if numGpus >= 1
utils/+helper/train_image2image_dag.m:  net.move('gpu') ;
utils/+helper/train_image2image_dag.m:      state.solverState{i} = gpuArray(s) ;
utils/+helper/train_image2image_dag.m:      state.solverState{i} = structfun(@gpuArray, s, 'UniformOutput', false) ;
utils/+helper/train_image2image_dag.m:if numGpus > 1
utils/+helper/train_image2image_dag.m:  if numGpus <= 1
utils/+helper/train_image2image_dag.m:  if numGpus <= 1
utils/+helper/train_image2image_dag.m:numGpus = numel(params.gpus) ;
utils/+helper/train_image2image_dag.m:otherGpus = setdiff(1:numGpus, labindex) ;
utils/+helper/train_image2image_dag.m:function prepareGPUs(opts, cold)
utils/+helper/train_image2image_dag.m:numGpus = numel(opts.gpus) ;
utils/+helper/train_image2image_dag.m:if numGpus > 1
utils/+helper/train_image2image_dag.m:  if ~isempty(pool) && pool.NumWorkers ~= numGpus
utils/+helper/train_image2image_dag.m:    parpool('local', numGpus) ;
utils/+helper/train_image2image_dag.m:if numGpus >= 1 && cold
utils/+helper/train_image2image_dag.m:  fprintf('%s: resetting GPU\n', mfilename)
utils/+helper/train_image2image_dag.m:  if numGpus == 1
utils/+helper/train_image2image_dag.m:    gpuDevice(opts.gpus)
utils/+helper/train_image2image_dag.m:      gpuDevice(opts.gpus(labindex))
utils/+helper/forward_wavresnet.m:    if opts.gpus > 0
utils/+helper/forward_wavresnet.m:        imageCoeffs = gpuArray(imageCoeffs);
utils/+helper/forward_wavresnet.m:        reconCoeffs = gpuArray(reconCoeffs);
utils/+helper/forward_wavresnet.m:        imagePatches = gpuArray(imagePatches);
utils/+helper/forward_wavresnet.m:            'mode','test', 'conserveMemory', 1, 'cudnn', opts.gpus > 0);
utils/+helper/forward_wavresnet.m:    if opts.gpus > 0
utils/+helper/train_image2image.m:%    The function supports training on CPU or on one or more GPUs
utils/+helper/train_image2image.m:%    (specify the list of GPU IDs in the `gpus` option).
utils/+helper/train_image2image.m:opts.gpus = [] ;
utils/+helper/train_image2image.m:  prepareGPUs(opts, epoch == start+1) ;
utils/+helper/train_image2image.m:  if numel(params.gpus) <= 1
utils/+helper/train_image2image.m:% With multiple GPUs, return one copy
utils/+helper/train_image2image.m:% move CNN  to GPU as needed
utils/+helper/train_image2image.m:numGpus = numel(params.gpus) ;
utils/+helper/train_image2image.m:if numGpus >= 1
utils/+helper/train_image2image.m:  net = vl_simplenn_move(net, 'gpu') ;
utils/+helper/train_image2image.m:        state.solverState{i}{j} = gpuArray(s) ;
utils/+helper/train_image2image.m:        state.solverState{i}{j} = structfun(@gpuArray, s, 'UniformOutput', false) ;
utils/+helper/train_image2image.m:if numGpus > 1
utils/+helper/train_image2image.m:  if numGpus <= 1
utils/+helper/train_image2image.m:    if numGpus >= 1
utils/+helper/train_image2image.m:      im = gpuArray(im) ;
utils/+helper/train_image2image.m:  if numGpus <= 1
utils/+helper/train_image2image.m:numGpus = numel(params.gpus) ;
utils/+helper/train_image2image.m:otherGpus = setdiff(1:numGpus, labindex) ;
utils/+helper/train_image2image.m:function prepareGPUs(params, cold)
utils/+helper/train_image2image.m:numGpus = numel(params.gpus) ;
utils/+helper/train_image2image.m:if numGpus > 1
utils/+helper/train_image2image.m:  if ~isempty(pool) && pool.NumWorkers ~= numGpus
utils/+helper/train_image2image.m:    parpool('local', numGpus) ;
utils/+helper/train_image2image.m:if numGpus >= 1 && cold
utils/+helper/train_image2image.m:  fprintf('%s: resetting GPU\n', mfilename) ;
utils/+helper/train_image2image.m:  if numGpus == 1
utils/+helper/train_image2image.m:    disp(gpuDevice(params.gpus)) ;
utils/+helper/train_image2image.m:      disp(gpuDevice(params.gpus(labindex))) ;

```
