# https://github.com/LiShuTJ/NDNet

```console
README.md:For TensorRT installation, please refer to [official installation guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).
README.md:2. For multi-GPU training, taking NDNet-DF1 as an example, type:
tools/predict.py:                        help='Use CPU to predict, use GPU by default.')
tools/predict.py:                        help='Use CPU to predict, use GPU by default.')
tools/predict.py:        model = model.cuda()
tools/predict.py:        image = image.cuda()
tools/test.py:    logger.info(get_model_summary(model.cuda(), dump_input.cuda()))
tools/test.py:    gpus = list(config.GPUS)
tools/test.py:    model = nn.DataParallel(model, device_ids=gpus).cuda()
tools/speedMeasure.py:    model = eval('models.'+config.MODEL.NAME)(19).cuda().eval()
tools/speedMeasure.py:    ).cuda()
tools/speedMeasure.py:    torch.cuda.synchronize()
tools/speedMeasure.py:    torch.cuda.synchronize()
tools/train.py:    gpus = list(config.GPUS)
tools/train.py:    distributed = len(gpus) > 1
tools/train.py:    device = torch.device('cuda:{}'.format(args.local_rank))
tools/train.py:        torch.cuda.set_device(args.local_rank)
tools/train.py:            backend="nccl", init_method="env://",
tools/train.py:        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
tools/train.py:            batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
tools/train.py:        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
tools/train.py:                        config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))
tools/exportONNX.py:    model = eval('models.'+config.MODEL.NAME)(19).cuda().eval()
tools/exportONNX.py:    ).cuda()
experiments/cityscapes/ndnet_res34_test.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_res34_test.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_res34_test.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_df2.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_df2.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_df2.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_res18_test.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_res18_test.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_res18_test.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_df1_test.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_df1_test.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_df1_test.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_res34.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_res34.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_res34.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_df1.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_df1.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_df1.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_df2_test.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_df2_test.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_df2_test.yaml:  BATCH_SIZE_PER_GPU: 4
experiments/cityscapes/ndnet_res18.yaml:GPUS: (0,1,2,3)
experiments/cityscapes/ndnet_res18.yaml:  BATCH_SIZE_PER_GPU: 3
experiments/cityscapes/ndnet_res18.yaml:  BATCH_SIZE_PER_GPU: 4
lib/datasets/cityscapes.py:                                        1.0865, 1.1529, 1.0507]).cuda()
lib/datasets/cityscapes.py:                                    ori_height,ori_width]).cuda()
lib/datasets/cityscapes.py:                                           new_h,new_w]).cuda()
lib/datasets/cityscapes.py:                count = torch.zeros([1,1, new_h, new_w]).cuda()
lib/datasets/lip.py:            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
lib/datasets/base_dataset.py:            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
lib/datasets/base_dataset.py:                                    ori_height,ori_width]).cuda()
lib/datasets/base_dataset.py:                                           new_h,new_w]).cuda()
lib/datasets/base_dataset.py:                count = torch.zeros([1,1, new_h, new_w]).cuda()
lib/config/default.py:_C.GPUS = (0,)
lib/config/default.py:_C.TRAIN.BATCH_SIZE_PER_GPU = 32
lib/config/default.py:_C.TEST.BATCH_SIZE_PER_GPU = 32
lib/models/sync_bn/inplace_abn/bn.py:    """InPlace Activated Batch Normalization with cross-GPU synchronization
lib/models/sync_bn/inplace_abn/bn.py:    This assumes that it will be replicated across GPUs using the same mechanism as in `nn.DataParallel`.
lib/models/sync_bn/inplace_abn/bn.py:            IDs of the GPUs that will run the replicas of this module.
lib/models/sync_bn/inplace_abn/bn.py:        self.devices = devices if devices else list(range(torch.cuda.device_count()))
lib/models/sync_bn/inplace_abn/functions.py:import torch.cuda.comm as comm
lib/models/sync_bn/inplace_abn/functions.py:                    "inplace_abn_cuda.cu"
lib/models/sync_bn/inplace_abn/functions.py:                extra_cuda_cflags=["--expt-extended-lambda"])
lib/models/sync_bn/inplace_abn/functions.py:        raise RuntimeError("CUDA Error encountered in {}".format(fn))
lib/models/sync_bn/inplace_abn/functions.py:            # TODO: implement simplified CUDA backward for inference mode
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (x.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return mean_var_cuda(x);
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (x.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return forward_cuda(x, mean, var, weight, bias, affine, eps);
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (z.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return edz_eydz_cuda(z, dz, weight, bias, affine, eps);
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (z.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return backward_cuda(z, dz, var, weight, bias, edz, eydz, affine, eps);
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (z.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return leaky_relu_backward_cuda(z, dz, slope);
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:  if (z.is_cuda()) {
lib/models/sync_bn/inplace_abn/src/inplace_abn.cpp:    return elu_backward_cuda(z, dz);
lib/models/sync_bn/inplace_abn/src/common.h:#include <cuda_runtime_api.h>
lib/models/sync_bn/inplace_abn/src/common.h:#if CUDART_VERSION >= 9000
lib/models/sync_bn/inplace_abn/src/common.h:#if __CUDA_ARCH__ >= 300
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:std::vector<at::Tensor> mean_var_cuda(at::Tensor x);
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:std::vector<at::Tensor> backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:void leaky_relu_backward_cuda(at::Tensor z, at::Tensor dz, float slope);
lib/models/sync_bn/inplace_abn/src/inplace_abn.h:void elu_backward_cuda(at::Tensor z, at::Tensor dz);
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:std::vector<at::Tensor> mean_var_cuda(at::Tensor x) {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(x.type(), "mean_var_cuda", ([&] {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:at::Tensor forward_cuda(at::Tensor x, at::Tensor mean, at::Tensor var, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(x.type(), "forward_cuda", ([&] {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:std::vector<at::Tensor> edz_eydz_cuda(at::Tensor z, at::Tensor dz, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(z.type(), "edz_eydz_cuda", ([&] {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:std::vector<at::Tensor> backward_cuda(at::Tensor z, at::Tensor dz, at::Tensor var, at::Tensor weight, at::Tensor bias,
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(z.type(), "backward_cuda", ([&] {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:void leaky_relu_backward_cuda(at::Tensor z, at::Tensor dz, float slope) {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(z.type(), "leaky_relu_backward_cuda", ([&] {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:void elu_backward_cuda(at::Tensor z, at::Tensor dz) {
lib/models/sync_bn/inplace_abn/src/inplace_abn_cuda.cu:  AT_DISPATCH_FLOATING_TYPES(z.type(), "leaky_relu_backward_cuda", ([&] {
lib/utils/utils.py:  Distribute the loss on multi-gpu to reduce 
lib/utils/utils.py:  the memory cost in the main gpu.

```
