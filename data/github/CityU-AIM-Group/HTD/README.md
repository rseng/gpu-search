# https://github.com/CityU-AIM-Group/HTD

```console
setup.py:                                       CUDAExtension)
setup.py:def make_cuda_ext(name, module, sources, sources_cuda=[]):
setup.py:    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
setup.py:        define_macros += [('WITH_CUDA', None)]
setup.py:        extension = CUDAExtension
setup.py:            '-D__CUDA_NO_HALF_OPERATORS__',
setup.py:            '-D__CUDA_NO_HALF_CONVERSIONS__',
setup.py:            '-D__CUDA_NO_HALF2_OPERATORS__',
setup.py:        sources += sources_cuda
setup.py:        print(f'Compiling {name} without CUDA')
configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py:    samples_per_gpu=4,
configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py:    workers_per_gpu=4,
configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py:    samples_per_gpu=2,
configs/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_4x2_2x_coco.py:    workers_per_gpu=2,
configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco.py:    samples_per_gpu=4,
configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_dcn_4x4_1x_coco.py:    workers_per_gpu=4,
configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    samples_per_gpu=4,
configs/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    workers_per_gpu=4,
configs/fcos/README.md:| R-50      | caffe   | N       | N        | N       | N       | 1x      | 5.2      | 22.9           | 36.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_4x4_1x_coco/fcos_r50_caffe_fpn_1x_4gpu_20200218-c229552f.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_4x4_1x_coco/20200224_230410.log.json) |
configs/fcos/README.md:| R-50      | caffe   | Y       | N        | N       | N       | 1x      | 6.5      | 22.7           | 36.6   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/fcos_r50_caffe_fpn_gn_1x_4gpu_20200218-7831950c.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco/20200130_004230.log.json) |
configs/fcos/README.md:| R-50      | caffe   | Y       | N        | N       | N       | 2x      | -        | -              | 36.9   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco/fcos_r50_caffe_fpn_gn_2x_4gpu_20200218-8ceb5c76.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r50_caffe_fpn_gn-head_4x4_2x_coco/20200130_004232.log.json) |
configs/fcos/README.md:| R-101     | caffe   | Y       | N        | N       | N       | 1x      | 10.2     | 17.3           | 39.2   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_4x4_1x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_1x_coco/fcos_r101_caffe_fpn_gn_1x_4gpu_20200218-13e2cc55.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_1x_coco/20200130_004231.log.json) |
configs/fcos/README.md:| R-101     | caffe   | Y       | N        | N       | N       | 2x      | -        | -              | 39.1   | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_2x_coco/fcos_r101_caffe_fpn_gn_2x_4gpu_20200218-d2261033.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_4x4_2x_coco/20200130_004231.log.json) |
configs/fcos/README.md:| R-101     | caffe   | Y       | Y        | 2x      | 10.2     | 17.3           | 40.9   |  [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py) | [model](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco/fcos_mstrain_640_800_r101_caffe_fpn_gn_2x_4gpu_20200218-d8a4f4cf.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco/20200130_004232.log.json) |
configs/fcos/README.md:- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU for R-50 and R-101 models, and 8 GPUs with 2 image/GPU for X-101 models.
configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py:    samples_per_gpu=4,
configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_4x4_2x_coco.py:    workers_per_gpu=4,
configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    samples_per_gpu=4,
configs/fcos/fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    workers_per_gpu=4,
configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:    samples_per_gpu=6,
configs/cornernet/cornernet_hourglass104_mstest_8x6_210e_coco.py:    workers_per_gpu=3,
configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:    samples_per_gpu=3,
configs/cornernet/cornernet_hourglass104_mstest_32x3_210e_coco.py:    workers_per_gpu=3,
configs/cornernet/README.md:- Experiments with `images_per_gpu=6` are conducted on Tesla V100-SXM2-32GB, `images_per_gpu=3` are conducted on GeForce GTX 1080 Ti.
configs/cornernet/README.md:    - 10 x 5: 10 GPUs with 5 images per gpu. This is the same setting as that reported in the original paper.
configs/cornernet/README.md:    - 8 x 6: 8 GPUs with 6 images per gpu. The total batchsize is similar to paper and only need 1 node to train.
configs/cornernet/README.md:    - 32 x 3: 32 GPUs with 3 images per gpu. The default setting for 1080TI and need 4 nodes to train.
configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:    samples_per_gpu=5,
configs/cornernet/cornernet_hourglass104_mstest_10x5_210e_coco.py:    workers_per_gpu=3,
configs/scratch/README.md:- The above models are trained with 16 GPUs.
configs/gfl/README.md:[4] *FPS is tested with a single GeForce RTX 2080Ti GPU, using a batch size of 1.*
configs/vfnet/vfnet_r50_fpn_1x_coco.py:    samples_per_gpu=2,
configs/vfnet/vfnet_r50_fpn_1x_coco.py:    workers_per_gpu=2,
configs/htd/htd_resnet101_dcn_2x_mstrain.py:    samples_per_gpu=2,
configs/htd/htd_resnet101_dcn_2x_mstrain.py:    workers_per_gpu=2,
configs/htd/htd_resnet101_dcn_2x_mstrain.py:dist_params = dict(backend='nccl')
configs/htd/htd_resnet101_dcn_2x_mstrain.py:gpu_ids = range(0, 6)
configs/htd/htd_resnetx101_dcn_2x_mstrain.py:    samples_per_gpu=2,
configs/htd/htd_resnetx101_dcn_2x_mstrain.py:    workers_per_gpu=2,
configs/htd/htd_resnetx101_dcn_2x_mstrain.py:dist_params = dict(backend='nccl')
configs/htd/htd_resnetx101_dcn_2x_mstrain.py:gpu_ids = range(0, 6)
configs/htd/htd_resnet101_2x_mstrain.py:    samples_per_gpu=2,
configs/htd/htd_resnet101_2x_mstrain.py:    workers_per_gpu=2,
configs/htd/htd_resnet101_2x_mstrain.py:dist_params = dict(backend='nccl')
configs/htd/htd_resnet101_2x_mstrain.py:gpu_ids = range(0, 6)
configs/htd/htd_resnet101_2x.py:    samples_per_gpu=2,
configs/htd/htd_resnet101_2x.py:    workers_per_gpu=2,
configs/htd/htd_resnet101_2x.py:dist_params = dict(backend='nccl')
configs/htd/htd_resnet101_2x.py:gpu_ids = range(0, 6)
configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    samples_per_gpu=4,
configs/nas_fcos/nas_fcos_fcoshead_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    workers_per_gpu=2,
configs/nas_fcos/README.md:- To be consistent with the author's implementation, we use 4 GPUs with 4 images/GPU.
configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    samples_per_gpu=4,
configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py:    workers_per_gpu=2,
configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py:    samples_per_gpu=8,
configs/nas_fpn/retinanet_r50_fpn_crop640_50e_coco.py:    workers_per_gpu=4,
configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py:    samples_per_gpu=8,
configs/nas_fpn/retinanet_r50_nasfpn_crop640_50e_coco.py:    workers_per_gpu=4,
configs/ssd/ssd512_coco.py:    samples_per_gpu=8,
configs/ssd/ssd512_coco.py:    workers_per_gpu=3,
configs/ssd/ssd300_coco.py:    samples_per_gpu=8,
configs/ssd/ssd300_coco.py:    workers_per_gpu=3,
configs/htc/README.md:- We use 8 GPUs with 2 images/GPU for R-50 and R-101 models, and 16 GPUs with 1 image/GPU for X-101 models.
configs/htc/README.md:If you would like to train X-101 HTC with 8 GPUs, you need to change the lr from 0.02 to 0.01.
configs/htc/htc_x101_64x4d_fpn_16x1_20e_coco.py:data = dict(samples_per_gpu=1, workers_per_gpu=1)
configs/htc/htc_x101_64x4d_fpn_dconv_c3-c5_mstrain_400_1400_16x1_20e_coco.py:    samples_per_gpu=1, workers_per_gpu=1, train=dict(pipeline=train_pipeline))
configs/htc/htc_x101_32x4d_fpn_16x1_20e_coco.py:data = dict(samples_per_gpu=1, workers_per_gpu=1)
configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py:    samples_per_gpu=6,
configs/centripetalnet/centripetalnet_hourglass104_mstest_16x6_210e_coco.py:    workers_per_gpu=3,
configs/legacy_1.x/ssd300_coco_v1.py:    samples_per_gpu=8,
configs/legacy_1.x/ssd300_coco_v1.py:    workers_per_gpu=3,
configs/legacy_1.x/ssd300_coco_v1.py:dist_params = dict(backend='nccl', port=29555)
configs/legacy_1.x/cascade_mask_rcnn_r50_fpn_1x_coco_v1.py:dist_params = dict(backend='nccl', port=29515)
configs/yolact/README.md:| Image Size | GPU x BS | Backbone      | *FPS  | mAP  | Weights | Configs | Download |
configs/yolact/README.md:All the aforementioned models are trained with a single GPU. It typically takes ~12GB VRAM when using resnet-101 as the backbone. If you want to try multiple GPUs training, you may have to modify the configuration files accordingly, such as adjusting the training schedule and freezing batch norm.
configs/yolact/README.md:# Trains using the resnet-101 backbone with a batch size of 8 on a single GPU.
configs/yolact/yolact_r50_1x8_coco.py:    samples_per_gpu=8,
configs/yolact/yolact_r50_1x8_coco.py:    workers_per_gpu=4,
configs/fsaf/README.md: - *All models are trained on 8 Titan-XP gpus and tested on a single gpu.*
configs/fast_rcnn/fast_rcnn_r50_fpn_1x_coco.py:    samples_per_gpu=2,
configs/fast_rcnn/fast_rcnn_r50_fpn_1x_coco.py:    workers_per_gpu=2,
configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco.py:    samples_per_gpu=2,
configs/mask_rcnn/mask_rcnn_x101_32x8d_fpn_1x_coco.py:    workers_per_gpu=2,
configs/free_anchor/README.md:- We use 8 GPUs with 2 images/GPU.
configs/pascal_voc/ssd300_voc0712.py:    samples_per_gpu=8,
configs/pascal_voc/ssd300_voc0712.py:    workers_per_gpu=3,
configs/detr/detr_r50_8x4_150e_coco.py:    samples_per_gpu=4,
configs/detr/detr_r50_8x4_150e_coco.py:    workers_per_gpu=4,
configs/carafe/README.md:The CUDA implementation of CARAFE can be find at https://github.com/myownskyW7/CARAFE.
configs/grid_rcnn/README.md:- All models are trained with 8 GPUs instead of 32 GPUs in the original paper.
configs/foveabox/README.md:[4] *We use 4 GPUs for training.*
configs/foveabox/fovea_r50_fpn_4x4_1x_coco.py:data = dict(samples_per_gpu=4, workers_per_gpu=4)
configs/cityscapes/README.md:- All baselines were trained using 8 GPU with a batch size of 8 (1 images per GPU) using the [linear scaling rule](https://arxiv.org/abs/1706.02677) to scale the learning rate.
configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_mstrain_2x.py:    samples_per_gpu=2,
configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_mstrain_2x.py:    workers_per_gpu=2,
configs/guided_anchoring/ga_retinanet_r101_caffe_fpn_mstrain_2x.py:dist_params = dict(backend='nccl')
configs/yolo/yolov3_d53_mstrain-608_273e_coco.py:    samples_per_gpu=8,
configs/yolo/yolov3_d53_mstrain-608_273e_coco.py:    workers_per_gpu=4,
configs/_base_/datasets/ead2019_detection.py:    samples_per_gpu=2,
configs/_base_/datasets/ead2019_detection.py:    workers_per_gpu=2,
configs/_base_/datasets/dhd_traffic.py:    samples_per_gpu=2,
configs/_base_/datasets/dhd_traffic.py:    workers_per_gpu=2,
configs/_base_/datasets/polyp.py:    samples_per_gpu=2,
configs/_base_/datasets/polyp.py:    workers_per_gpu=2,
configs/_base_/datasets/cityscapes_instance.py:    samples_per_gpu=1,
configs/_base_/datasets/cityscapes_instance.py:    workers_per_gpu=2,
configs/_base_/datasets/polyp_detection.py:    samples_per_gpu=2,
configs/_base_/datasets/polyp_detection.py:    workers_per_gpu=2,
configs/_base_/datasets/coco_detection.py:    samples_per_gpu=2,
configs/_base_/datasets/coco_detection.py:    workers_per_gpu=2,
configs/_base_/datasets/deepfashion.py:    imgs_per_gpu=2,
configs/_base_/datasets/deepfashion.py:    workers_per_gpu=1,
configs/_base_/datasets/lvis_v0.5_instance.py:    samples_per_gpu=2,
configs/_base_/datasets/lvis_v0.5_instance.py:    workers_per_gpu=2,
configs/_base_/datasets/cityscapes_detection.py:    samples_per_gpu=1,
configs/_base_/datasets/cityscapes_detection.py:    workers_per_gpu=2,
configs/_base_/datasets/coco_instance_semantic.py:    samples_per_gpu=2,
configs/_base_/datasets/coco_instance_semantic.py:    workers_per_gpu=2,
configs/_base_/datasets/coco_instance.py:    samples_per_gpu=2,
configs/_base_/datasets/coco_instance.py:    workers_per_gpu=2,
configs/_base_/datasets/lvis_v1_instance.py:    samples_per_gpu=2,
configs/_base_/datasets/lvis_v1_instance.py:    workers_per_gpu=2,
configs/_base_/datasets/wider_face.py:    samples_per_gpu=60,
configs/_base_/datasets/wider_face.py:    workers_per_gpu=2,
configs/_base_/datasets/voc0712.py:    samples_per_gpu=2,
configs/_base_/datasets/voc0712.py:    workers_per_gpu=2,
configs/_base_/default_runtime.py:dist_params = dict(backend='nccl')
docs/projects.md:- Overcoming Classifier Imbalance for Long-tail Object Detection with Balanced Group Softmax, CVPR2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Overcoming_Classifier_Imbalance_for_Long-Tail_Object_Detection_With_Balanced_Group_CVPR_2020_paper.pdf)[[github]](https://github.com/FishYuLi/BalancedGroupSoftmax)
docs/projects.md:- Look-into-Object: Self-supervised Structure Modeling for Object Recognition, CVPR 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_Look-Into-Object_Self-Supervised_Structure_Modeling_for_Object_Recognition_CVPR_2020_paper.pdf)[[github]](https://github.com/JDAI-CV/LIO)
docs/projects.md:- D2Det: Towards High Quality Object Detection and Instance Segmentation, CVPR2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Cao_D2Det_Towards_High_Quality_Object_Detection_and_Instance_Segmentation_CVPR_2020_paper.html)[[github]](https://github.com/JialeCao001/D2Det)
docs/projects.md:- Learning a Unified Sample Weighting Network for Object Detection, CVPR 2020. [[paper]](http://openaccess.thecvf.com/content_CVPR_2020/html/Cai_Learning_a_Unified_Sample_Weighting_Network_for_Object_Detection_CVPR_2020_paper.html)[[github]](https://github.com/caiqi/sample-weighting-network)
docs/projects.md:- Reasoning R-CNN: Unifying Adaptive Global Reasoning into Large-scale Object Detection, CVPR2019. [[paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Reasoning-RCNN_Unifying_Adaptive_Global_Reasoning_Into_Large-Scale_Object_Detection_CVPR_2019_paper.pdf)[[github]](https://github.com/chanyn/Reasoning-RCNN)
docs/robustness_benchmarking.md:- [x] single GPU testing
docs/robustness_benchmarking.md:- [ ] multiple GPU testing
docs/robustness_benchmarking.md:# single-gpu testing
docs/tutorials/customize_dataset.md:    samples_per_gpu=2,
docs/tutorials/customize_dataset.md:    workers_per_gpu=2,
docs/tutorials/customize_dataset.md:        imgs_per_gpu=2,
docs/tutorials/customize_dataset.md:        workers_per_gpu=2,
docs/tutorials/customize_dataset.md:        imgs_per_gpu=2,
docs/tutorials/customize_dataset.md:        workers_per_gpu=2,
docs/tutorials/customize_dataset.md:    imgs_per_gpu=2,
docs/tutorials/customize_dataset.md:    workers_per_gpu=2,
docs/tutorials/config.md:{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
docs/tutorials/config.md:- `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
docs/tutorials/config.md:    samples_per_gpu=2,  # Batch size of a single GPU
docs/tutorials/config.md:    workers_per_gpu=2,  # Worker to pre-fetch data for each single GPU
docs/tutorials/config.md:        samples_per_gpu=2  # Batch size of a single GPU used in testing
docs/tutorials/config.md:dist_params = dict(backend='nccl')  # Parameters to setup distributed training, the port can also be set.
docs/model_zoo.md:- For fair comparison with other codebases, we report the GPU memory as the maximum value of `torch.cuda.max_memory_allocated()` for all 8 GPUs. Note that this value is usually less than what `nvidia-smi` shows.
docs/model_zoo.md:We also provide the [checkpoint](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_compare_20200518-10127928.pth) and [training log](http://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_20200518_105755.log.json) for reference. The throughput is computed as the average throughput in iterations 100-500 to skip GPU warmup time.
docs/model_zoo.md:- 8 NVIDIA Tesla V100 (32G) GPUs
docs/model_zoo.md:- CUDA 10.1
docs/model_zoo.md:- NCCL 2.4.08
docs/model_zoo.md:The inference speed is measured with fps (img/s) on a single GPU, the higher, the better.
docs/1_exist_data_model.md:model = init_detector(config_file, checkpoint_file, device='cuda:0')
docs/1_exist_data_model.md:By utilizing CUDA streams, it allows not to block CPU on GPU bound inference code and enables better CPU/GPU utilization for single-threaded application. Inference can be done concurrently either between different input data samples or between different models of some inference pipeline.
docs/1_exist_data_model.md:    device = 'cuda:0'
docs/1_exist_data_model.md:        streamqueue.put_nowait(torch.cuda.Stream(device=device))
docs/1_exist_data_model.md:    [--device ${GPU_ID}] \
docs/1_exist_data_model.md:    [--device ${GPU_ID}] \
docs/1_exist_data_model.md:- single GPU
docs/1_exist_data_model.md:- single node multiple GPUs
docs/1_exist_data_model.md:# single-gpu testing
docs/1_exist_data_model.md:# multi-gpu testing
docs/1_exist_data_model.md:    ${GPU_NUM} \
docs/1_exist_data_model.md:- `--show`: If specified, detection results will be plotted on the images and shown in a new window. It is only applicable to single GPU testing and used for debugging and visualization. Please make sure that GUI is available in your environment. Otherwise, you may encounter an error like `cannot connect to X server`.
docs/1_exist_data_model.md:- `--show-dir`: If specified, detection results will be plotted on the images and saved to the specified directory. It is only applicable to single GPU testing and used for debugging and visualization. You do NOT need a GUI available in your environment for using this option.
docs/1_exist_data_model.md:4. Test Mask R-CNN with 8 GPUs, and evaluate the bbox and mask AP.
docs/1_exist_data_model.md:5. Test Mask R-CNN with 8 GPUs, and evaluate the **classwise** bbox and mask AP.
docs/1_exist_data_model.md:6. Test Mask R-CNN on COCO test-dev with 8 GPUs, and generate JSON files for submitting to the official evaluation server.
docs/1_exist_data_model.md:7. Test Mask R-CNN on Cityscapes test with 8 GPUs, and generate txt and png files for submitting to the official evaluation server.
docs/1_exist_data_model.md:**Important**: The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8\*2 = 16).
docs/1_exist_data_model.md:According to the [linear scaling rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., `lr=0.01` for 4 GPUs \* 2 imgs/gpu and `lr=0.08` for 16 GPUs \* 4 imgs/gpu.
docs/1_exist_data_model.md:### Training on a single GPU
docs/1_exist_data_model.md:We provide `tools/train.py` to launch training jobs on a single GPU.
docs/1_exist_data_model.md:### Training on multiple GPUs
docs/1_exist_data_model.md:We provide `tools/dist_train.sh` to launch training on multiple GPUs.
docs/1_exist_data_model.md:    ${GPU_NUM} \
docs/1_exist_data_model.md:Optional arguments remain the same as stated [above](#train-with-a-single-GPU).
docs/1_exist_data_model.md:If you would like to launch multiple jobs on a single machine, e.g., 2 jobs of 4-GPU training on a machine with 8 GPUs,
docs/1_exist_data_model.md:CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./tools/dist_train.sh ${CONFIG_FILE} 4
docs/1_exist_data_model.md:CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=29501 ./tools/dist_train.sh ${CONFIG_FILE} 4
docs/1_exist_data_model.md:[GPUS=${GPUS}] ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
docs/1_exist_data_model.md:Below is an example of using 16 GPUs to train Mask R-CNN on a Slurm partition named _dev_, and set the work-dir to some shared file systems.
docs/1_exist_data_model.md:GPUS=16 ./tools/slurm_train.sh dev mask_r50_1x configs/mask_rcnn_r50_fpn_1x_coco.py /nfs/xxxx/mask_rcnn_r50_fpn_1x
docs/1_exist_data_model.md:   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR} --options 'dist_params.port=29500'
docs/1_exist_data_model.md:   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR} --options 'dist_params.port=29501'
docs/1_exist_data_model.md:   dist_params = dict(backend='nccl', port=29500)
docs/1_exist_data_model.md:   dist_params = dict(backend='nccl', port=29501)
docs/1_exist_data_model.md:   CUDA_VISIBLE_DEVICES=0,1,2,3 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config1.py ${WORK_DIR}
docs/1_exist_data_model.md:   CUDA_VISIBLE_DEVICES=4,5,6,7 GPUS=4 ./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} config2.py ${WORK_DIR}
docs/faq.md:## PyTorch/CUDA Environment
docs/faq.md:    1. Temporary work-around: do `MMCV_WITH_OPS=1 MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80' pip install -e .`.
docs/faq.md:    The common issue is `nvcc fatal : Unsupported gpu architecture 'compute_86'`. This means that the compiler should optimize for sm_86, i.e., nvidia 30 series card, but such optimizations have not been supported by CUDA toolkit 11.0.
docs/faq.md:    This work-around modifies the compile flag by adding `MMCV_CUDA_ARGS='-gencode=arch=compute_80,code=sm_80'`, which tells `nvcc` to optimize for **sm_80**, i.e., Nvidia A100. Although A100 is different from the 30 series card, they use similar ampere architecture. This may hurt the performance but it works.
docs/faq.md:    1. Check if your cuda runtime version (under `/usr/local/`), `nvcc --version` and `conda list cudatoolkit` version match.
docs/faq.md:    2. Run `python mmdet/utils/collect_env.py` to check whether PyTorch, torchvision, and MMCV are built for the correct GPU architecture.
docs/faq.md:    You may need to set `TORCH_CUDA_ARCH_LIST` to reinstall MMCV.
docs/faq.md:    The GPU arch table could be found [here](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#gpu-feature-list),
docs/faq.md:    i.e. run `TORCH_CUDA_ARCH_LIST=7.0 pip install mmcv-full` to build MMCV for Volta GPUs.
docs/faq.md:    The compatibility issue could happen when using old GPUS, e.g., Tesla K80 (3.7) on colab.
docs/faq.md:    For example, you may compile mmcv using CUDA 10.0 but run it on CUDA 9.0 environments.
docs/faq.md:    1. If those symbols are CUDA/C++ symbols (e.g., libcudart.so or GLIBCXX), check whether the CUDA/GCC runtimes are the same as those used for compiling mmcv,
docs/faq.md:    i.e. run `python mmdet/utils/collect_env.py` to see if `"MMCV Compiler"`/`"MMCV CUDA Compiler"` is the same as `"GCC"`/`"CUDA_HOME"`.
docs/faq.md:    2. Check whether PyTorch is correctly installed and could use CUDA op, e.g. type the following command in your terminal.
docs/faq.md:        python -c 'import torch; print(torch.cuda.is_available())'
docs/faq.md:- ’GPU out of memory"
docs/faq.md:  You can set `gpu_assign_thr=N` in the config of assigner thus the assigner will calculate box overlaps through CPU when there are more than N GT boxes.
docs/faq.md:  2. Set `with_cp=True` in the backbone. This uses the sublinear strategy in PyTorch to reduce GPU memory cost in the backbone.
docs/compatibility.md:  [model]_(model setting)_[backbone]_[neck]_(norm setting)_(misc)_(gpu x batch)_[schedule]_[dataset].py,
docs/changelog.md:- Fix bug of `gpu_id` in distributed training mode (#4163)
docs/changelog.md:- Add solution to installation issues in 30-series GPUs (#4176)
docs/changelog.md:- Support Batch Inference (#3564, #3686, #3705): Since v2.4.0, MMDetection could inference model with multiple images in a single GPU.
docs/changelog.md:- Fix atss when sampler per gpu is 1 (#3528)
docs/changelog.md:- The CUDA/C++ operators have been moved to `mmcv.ops`. For backward compatibility `mmdet.ops` is kept as warppers of `mmcv.ops`.
docs/changelog.md:- Move CUDA/C++ operators into `mmcv.ops` and keep `mmdet.ops` as warppers for backward compatibility (#3232)(#3457)
docs/changelog.md:- Add warnings when deprecated `imgs_per_gpu` is used. (#2700)
docs/changelog.md:- Unify cuda and cpp API for custom ops. (#2277)
docs/changelog.md:- Fix the inference demo on devices other than gpu:0. (#2098)
docs/changelog.md:- Fix the NMS issue on devices other than GPU:0. (#1603)
docs/changelog.md:- Fix zero outputs in DeformConv when not running on cuda:0. (#1326)
docs/changelog.md:- Update the PyTorch and CUDA version in the docker file. (#1615)
docs/changelog.md:- Use int64_t instead of long in cuda kernels. (#1131)
docs/changelog.md:- Speed up multi-gpu testing.
docs/changelog.md:- Replace NMS and SigmoidFocalLoss with Pytorch CUDA extensions.
docs/get_started.md:- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
docs/get_started.md:    Note: Make sure that your compilation CUDA version and runtime CUDA version match.
docs/get_started.md:    You can check the supported CUDA version for precompiled packages on the [PyTorch website](https://pytorch.org/).
docs/get_started.md:    `E.g.1` If you have CUDA 10.1 installed under `/usr/local/cuda` and would like to install
docs/get_started.md:    PyTorch 1.5, you need to install the prebuilt PyTorch with CUDA 10.1.
docs/get_started.md:    conda install pytorch cudatoolkit=10.1 torchvision -c pytorch
docs/get_started.md:    `E.g. 2` If you have CUDA 9.2 installed under `/usr/local/cuda` and would like to install
docs/get_started.md:    PyTorch 1.3.1., you need to install the prebuilt PyTorch with CUDA 9.2.
docs/get_started.md:    conda install pytorch=1.3.1 cudatoolkit=9.2 torchvision=0.4.2 -c pytorch
docs/get_started.md:    you can use more CUDA versions such as 9.0.
docs/get_started.md:    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.
docs/get_started.md:The code can be built for CPU only environment (where CUDA isn't available).
docs/get_started.md:- nms_cuda
docs/get_started.md:- sigmoid_focal_loss_cuda
docs/get_started.md:# build an image with PyTorch 1.6, CUDA 10.1
docs/get_started.md:docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
docs/get_started.md:Assuming that you already have CUDA 10.1 installed, here is a full script for setting up MMDetection with conda.
docs/get_started.md:conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
docs/get_started.md:device = 'cuda:0'
tests/test_data/test_dataset.py:@patch('mmdet.apis.single_gpu_test', MagicMock)
tests/test_data/test_dataset.py:@patch('mmdet.apis.multi_gpu_test', MagicMock)
tests/async_benchmark.py:    Sample runs for 20 demo images on K80 GPU, model - mask_rcnn_r50_fpn_1x:
tests/async_benchmark.py:    device = 'cuda:0'
tests/async_benchmark.py:        streamqueue.put_nowait(torch.cuda.Stream(device=device))
tests/async_benchmark.py:    with torch.cuda.stream(torch.cuda.default_stream()):
tests/test_eval_hook.py:    not torch.cuda.is_available(), reason='requires CUDA support')
tests/test_eval_hook.py:@patch('mmdet.apis.single_gpu_test', MagicMock)
tests/test_eval_hook.py:@patch('mmdet.apis.multi_gpu_test', MagicMock)
tests/test_models/test_forward.py:def test_single_stage_forward_gpu(cfg_file):
tests/test_models/test_forward.py:    if not torch.cuda.is_available():
tests/test_models/test_forward.py:        pytest.skip('test requires GPU and torch+cuda')
tests/test_models/test_forward.py:    detector = detector.cuda()
tests/test_models/test_forward.py:    imgs = imgs.cuda()
tests/test_models/test_forward.py:    gt_bboxes = [b.cuda() for b in mm_inputs['gt_bboxes']]
tests/test_models/test_forward.py:    gt_labels = [g.cuda() for g in mm_inputs['gt_labels']]
tests/test_models/test_heads.py:    if torch.cuda.is_available():
tests/test_models/test_heads.py:        self.cuda()
tests/test_models/test_heads.py:            torch.rand(1, 1, s // feat_size, s // feat_size).cuda()
tests/test_models/test_heads.py:        gt_bboxes = [torch.empty((0, 4)).cuda()]
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([]).cuda()]
tests/test_models/test_heads.py:            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([2]).cuda()]
tests/test_models/test_heads.py:    if torch.cuda.is_available():
tests/test_models/test_heads.py:        head.cuda()
tests/test_models/test_heads.py:            torch.rand(1, 1, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
tests/test_models/test_heads.py:            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([2]).cuda()]
tests/test_models/test_heads.py:        gt_bboxes = [torch.empty((0, 4)).cuda()]
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([]).cuda()]
tests/test_models/test_heads.py:    if torch.cuda.is_available():
tests/test_models/test_heads.py:        head.cuda()
tests/test_models/test_heads.py:            torch.rand(1, 4, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
tests/test_models/test_heads.py:        gt_bboxes = [torch.empty((0, 4)).cuda()]
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([]).cuda()]
tests/test_models/test_heads.py:            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([2]).cuda()]
tests/test_models/test_heads.py:    if torch.cuda.is_available():
tests/test_models/test_heads.py:        head.cuda()
tests/test_models/test_heads.py:            torch.rand(1, 3, s // (2**(i + 2)), s // (2**(i + 2))).cuda()
tests/test_models/test_heads.py:        gt_bboxes = [torch.empty((0, 4)).cuda()]
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([]).cuda()]
tests/test_models/test_heads.py:            torch.Tensor([[23.6667, 23.8757, 238.6326, 151.8874]]).cuda(),
tests/test_models/test_heads.py:        gt_labels = [torch.LongTensor([2]).cuda()]
tests/test_async.py:                 device='cuda:0'):
tests/test_async.py:            stream = torch.cuda.Stream(device=self.device)
tests/test_async.py:            if not torch.cuda.is_available():
tests/test_async.py:                pytest.skip('test requires GPU and torch+cuda')
tests/test_anchor.py:    if torch.cuda.is_available():
tests/test_anchor.py:        device = 'cuda'
tests/test_anchor.py:    if torch.cuda.is_available():
tests/test_anchor.py:        device = 'cuda'
tests/test_anchor.py:    if torch.cuda.is_available():
tests/test_anchor.py:        device = 'cuda'
tests/test_anchor.py:    if torch.cuda.is_available():
tests/test_anchor.py:        device = 'cuda'
tests/test_anchor.py:    if torch.cuda.is_available():
tests/test_anchor.py:        device = 'cuda'
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:        output_x, output_y = model(input_x.cuda(), input_y.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:        output_x, output_y = model(input_x.cuda(), input_y.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:        output_x, output_y = model(input_x.cuda(), input_y.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:        output_x, output_y = model(input_x.cuda(), input_y.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
tests/test_fp16.py:    if torch.cuda.is_available():
tests/test_fp16.py:        model.cuda()
tests/test_fp16.py:            input_x.cuda(), y=input_y.cuda(), z=input_z.cuda())
mmdet/datasets/samplers/group_sampler.py:    def __init__(self, dataset, samples_per_gpu=1):
mmdet/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
mmdet/datasets/samplers/group_sampler.py:                size / self.samples_per_gpu)) * self.samples_per_gpu
mmdet/datasets/samplers/group_sampler.py:            num_extra = int(np.ceil(size / self.samples_per_gpu)
mmdet/datasets/samplers/group_sampler.py:                            ) * self.samples_per_gpu - len(indice)
mmdet/datasets/samplers/group_sampler.py:            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
mmdet/datasets/samplers/group_sampler.py:                range(len(indices) // self.samples_per_gpu))
mmdet/datasets/samplers/group_sampler.py:                 samples_per_gpu=1,
mmdet/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
mmdet/datasets/samplers/group_sampler.py:                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
mmdet/datasets/samplers/group_sampler.py:                          self.num_replicas)) * self.samples_per_gpu
mmdet/datasets/samplers/group_sampler.py:                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
mmdet/datasets/samplers/group_sampler.py:                ) * self.samples_per_gpu * self.num_replicas - len(indice)
mmdet/datasets/samplers/group_sampler.py:                    len(indices) // self.samples_per_gpu, generator=g))
mmdet/datasets/samplers/group_sampler.py:            for j in range(i * self.samples_per_gpu, (i + 1) *
mmdet/datasets/samplers/group_sampler.py:                           self.samples_per_gpu)
mmdet/datasets/builder.py:                     samples_per_gpu,
mmdet/datasets/builder.py:                     workers_per_gpu,
mmdet/datasets/builder.py:                     num_gpus=1,
mmdet/datasets/builder.py:    In distributed training, each GPU/process has a dataloader.
mmdet/datasets/builder.py:    In non-distributed training, there is only one dataloader for all GPUs.
mmdet/datasets/builder.py:        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
mmdet/datasets/builder.py:            batch size of each GPU.
mmdet/datasets/builder.py:        workers_per_gpu (int): How many subprocesses to use for data loading
mmdet/datasets/builder.py:            for each GPU.
mmdet/datasets/builder.py:        num_gpus (int): Number of GPUs. Only used in non-distributed training.
mmdet/datasets/builder.py:        # that images on each GPU are in the same group
mmdet/datasets/builder.py:            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
mmdet/datasets/builder.py:        batch_size = samples_per_gpu
mmdet/datasets/builder.py:        num_workers = workers_per_gpu
mmdet/datasets/builder.py:        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
mmdet/datasets/builder.py:        batch_size = num_gpus * samples_per_gpu
mmdet/datasets/builder.py:        num_workers = num_gpus * workers_per_gpu
mmdet/datasets/builder.py:        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
mmdet/models/detectors/base.py:        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
mmdet/models/detectors/base.py:        samples_per_gpu = img[0].size(0)
mmdet/models/detectors/base.py:        assert samples_per_gpu == 1
mmdet/models/detectors/base.py:                DDP, it means the batch size on each GPU), which is used for \
mmdet/models/losses/focal_loss.py:    r"""A warpper of cuda version `Focal Loss
mmdet/models/dense_heads/anchor_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
mmdet/models/dense_heads/vfnet_head.py:            examples across GPUs. Default: True
mmdet/models/dense_heads/vfnet_head.py:        # sync num_pos across all gpus
mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = reduce_mean(
mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = num_pos
mmdet/models/dense_heads/vfnet_head.py:            iou_targets_ini_avg_per_gpu = reduce_mean(
mmdet/models/dense_heads/vfnet_head.py:            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
mmdet/models/dense_heads/vfnet_head.py:            iou_targets_rf_avg_per_gpu = reduce_mean(
mmdet/models/dense_heads/vfnet_head.py:            bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)
mmdet/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
mmdet/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
mmdet/models/dense_heads/transformer_head.py:        # Compute the average number of gt boxes accross all gpus, for
mmdet/models/dense_heads/sabl_retina_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
mmdet/models/dense_heads/guided_anchor_head.py:    def get_sampled_approxs(self, featmap_sizes, img_metas, device='cuda'):
mmdet/models/dense_heads/guided_anchor_head.py:                    device='cuda'):
mmdet/models/dense_heads/guided_anchor_head.py:        img_per_gpu = len(gt_bboxes_list)
mmdet/models/dense_heads/guided_anchor_head.py:                img_per_gpu,
mmdet/models/dense_heads/guided_anchor_head.py:        for img_id in range(img_per_gpu):
mmdet/models/dense_heads/gfl_head.py:                reduced over all GPUs.
mmdet/models/dense_heads/gfl_head.py:            weight_targets = torch.tensor(0).cuda()
mmdet/models/dense_heads/atss_head.py:                reduced over all GPUs.
mmdet/models/dense_heads/atss_head.py:            torch.tensor(num_total_pos).cuda()).item()
mmdet/models/dense_heads/rpn_test_mixin.py:        samples_per_gpu = len(img_metas[0])
mmdet/models/dense_heads/rpn_test_mixin.py:        aug_proposals = [[] for _ in range(samples_per_gpu)]
mmdet/models/dense_heads/rpn_test_mixin.py:        for i in range(samples_per_gpu):
mmdet/models/roi_heads/bbox_heads/bbox_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
mmdet/models/roi_heads/bbox_heads/sabl_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
mmdet/models/roi_heads/mask_heads/d2det_head.py:        targets = targets.cuda()
mmdet/models/roi_heads/mask_heads/d2det_head.py:        points = points.cuda()
mmdet/models/roi_heads/mask_heads/d2det_head.py:        masks = masks.cuda()
mmdet/models/roi_heads/mask_heads/d2det_head.py:        idx = (torch.arange(0, map_size).float() + 0.5).cuda() / map_size
mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:            # GPU benefits from parallelism for larger chunks,
mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:    # On GPU, paste all masks together (up to chunk size)
mmdet/core/evaluation/eval_hooks.py:        from mmdet.apis import single_gpu_test
mmdet/core/evaluation/eval_hooks.py:        results = single_gpu_test(runner.model, self.dataloader, show=False)
mmdet/core/evaluation/eval_hooks.py:        gpu_collect (bool): Whether to use gpu or cpu to collect results.
mmdet/core/evaluation/eval_hooks.py:                 gpu_collect=False,
mmdet/core/evaluation/eval_hooks.py:        self.gpu_collect = gpu_collect
mmdet/core/evaluation/eval_hooks.py:        from mmdet.apis import multi_gpu_test
mmdet/core/evaluation/eval_hooks.py:        results = multi_gpu_test(
mmdet/core/evaluation/eval_hooks.py:            gpu_collect=self.gpu_collect)
mmdet/core/post_processing/bbox_nms.py:    implement Fast NMS entirely in standard GPU-accelerated matrix operations.
mmdet/core/anchor/anchor_generator.py:    def grid_anchors(self, featmap_sizes, device='cuda'):
mmdet/core/anchor/anchor_generator.py:                                  device='cuda'):
mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
mmdet/core/anchor/anchor_generator.py:    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
mmdet/core/anchor/anchor_generator.py:                                 device='cuda'):
mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
mmdet/core/anchor/anchor_generator.py:    def responsible_flags(self, featmap_sizes, gt_bboxes, device='cuda'):
mmdet/core/anchor/anchor_generator.py:                                       device='cuda'):
mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
mmdet/core/anchor/point_generator.py:    def grid_points(self, featmap_size, stride=16, device='cuda'):
mmdet/core/anchor/point_generator.py:    def valid_flags(self, featmap_size, valid_size, device='cuda'):
mmdet/core/bbox/assigners/max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
mmdet/core/bbox/assigners/max_iou_assigner.py:                 gpu_assign_thr=-1,
mmdet/core/bbox/assigners/max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
mmdet/core/bbox/assigners/max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
mmdet/core/bbox/assigners/max_iou_assigner.py:            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
mmdet/core/bbox/assigners/approx_max_iou_assigner.py:                 gpu_assign_thr=-1,
mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
mmdet/core/bbox/assigners/approx_max_iou_assigner.py:            num_gts > self.gpu_assign_thr) else False
mmdet/core/bbox/samplers/random_sampler.py:            if torch.cuda.is_available():
mmdet/core/bbox/samplers/random_sampler.py:                device = torch.cuda.current_device()
mmdet/core/bbox/samplers/sampling_result.py:            >>> # xdoctest: +REQUIRES(--gpu)
mmdet/core/bbox/samplers/score_hlr_sampler.py:            if torch.cuda.is_available():
mmdet/core/bbox/samplers/score_hlr_sampler.py:                device = torch.cuda.current_device()
mmdet/core/utils/dist_utils.py:    """"Obtain the mean of tensor on different GPUs."""
mmdet/ops/__init__.py:                      get_compiling_cuda_version, modulated_deform_conv, nms,
mmdet/ops/__init__.py:    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
mmdet/utils/contextmanagers.py:                    streams: List[torch.cuda.Stream] = None):
mmdet/utils/contextmanagers.py:    """Async context manager that waits for work to complete on given CUDA
mmdet/utils/contextmanagers.py:    if not torch.cuda.is_available():
mmdet/utils/contextmanagers.py:    stream_before_context_switch = torch.cuda.current_stream()
mmdet/utils/contextmanagers.py:        torch.cuda.Event(enable_timing=DEBUG_COMPLETED_TIME) for _ in streams
mmdet/utils/contextmanagers.py:        start = torch.cuda.Event(enable_timing=True)
mmdet/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
mmdet/utils/contextmanagers.py:        with torch.cuda.stream(stream_before_context_switch):
mmdet/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
mmdet/utils/contextmanagers.py:    if not torch.cuda.is_available():
mmdet/utils/contextmanagers.py:    initial_stream = torch.cuda.current_stream()
mmdet/utils/contextmanagers.py:    with torch.cuda.stream(initial_stream):
mmdet/utils/contextmanagers.py:        assert isinstance(stream, torch.cuda.Stream)
mmdet/utils/contextmanagers.py:            with torch.cuda.stream(stream):
mmdet/utils/contextmanagers.py:                current = torch.cuda.current_stream()
mmdet/utils/profiling.py:        """Print time spent by CPU and GPU.
mmdet/utils/profiling.py:        if (not enabled) or not torch.cuda.is_available():
mmdet/utils/profiling.py:        stream = stream if stream else torch.cuda.current_stream()
mmdet/utils/profiling.py:        start = torch.cuda.Event(enable_timing=True)
mmdet/utils/profiling.py:        end = torch.cuda.Event(enable_timing=True)
mmdet/utils/profiling.py:            gpu_time = start.elapsed_time(end)
mmdet/utils/profiling.py:            msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
mmdet/apis/test.py:def single_gpu_test(model,
mmdet/apis/test.py:def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
mmdet/apis/test.py:    """Test model with multiple gpus.
mmdet/apis/test.py:    This method tests model with multiple gpus and collects the results
mmdet/apis/test.py:    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
mmdet/apis/test.py:    it encodes results to gpu tensors and use gpu communication for results
mmdet/apis/test.py:    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
mmdet/apis/test.py:            different gpus under cpu mode.
mmdet/apis/test.py:        gpu_collect (bool): Option to use either gpu or cpu to collect results.
mmdet/apis/test.py:    if gpu_collect:
mmdet/apis/test.py:        results = collect_results_gpu(results, len(dataset))
mmdet/apis/test.py:                                device='cuda')
mmdet/apis/test.py:                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
mmdet/apis/test.py:def collect_results_gpu(result_part, size):
mmdet/apis/test.py:        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
mmdet/apis/test.py:    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
mmdet/apis/test.py:    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
mmdet/apis/inference.py:def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
mmdet/apis/inference.py:    data = collate([data], samples_per_gpu=1)
mmdet/apis/inference.py:    if next(model.parameters()).is_cuda:
mmdet/apis/inference.py:        # scatter to specified GPU
mmdet/apis/inference.py:    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
mmdet/apis/train.py:    torch.cuda.manual_seed_all(seed)
mmdet/apis/train.py:    if 'imgs_per_gpu' in cfg.data:
mmdet/apis/train.py:        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
mmdet/apis/train.py:                       'Please use "samples_per_gpu" instead')
mmdet/apis/train.py:        if 'samples_per_gpu' in cfg.data:
mmdet/apis/train.py:                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
mmdet/apis/train.py:                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
mmdet/apis/train.py:                f'={cfg.data.imgs_per_gpu} is used in this experiments')
mmdet/apis/train.py:                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
mmdet/apis/train.py:                f'{cfg.data.imgs_per_gpu} in this experiments')
mmdet/apis/train.py:        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
mmdet/apis/train.py:            cfg.data.samples_per_gpu,
mmdet/apis/train.py:            cfg.data.workers_per_gpu,
mmdet/apis/train.py:            # cfg.gpus will be ignored if distributed
mmdet/apis/train.py:            len(cfg.gpu_ids),
mmdet/apis/train.py:    # put model on gpus
mmdet/apis/train.py:            model.cuda(),
mmdet/apis/train.py:            device_ids=[torch.cuda.current_device()],
mmdet/apis/train.py:            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
mmdet/apis/train.py:        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
mmdet/apis/train.py:        if val_samples_per_gpu > 1:
mmdet/apis/train.py:            samples_per_gpu=val_samples_per_gpu,
mmdet/apis/train.py:            workers_per_gpu=cfg.data.workers_per_gpu,
mmdet/apis/__init__.py:from .test import multi_gpu_test, single_gpu_test
mmdet/apis/__init__.py:    'multi_gpu_test', 'single_gpu_test'
README.md:cudatoolkit: 10.0
tools/slurm_train.sh:GPUS=${GPUS:-8}
tools/slurm_train.sh:GPUS_PER_NODE=${GPUS_PER_NODE:-8}
tools/slurm_train.sh:    --gres=gpu:${GPUS_PER_NODE} \
tools/slurm_train.sh:    --ntasks=${GPUS} \
tools/slurm_train.sh:    --ntasks-per-node=${GPUS_PER_NODE} \
tools/get_flops.py:    if torch.cuda.is_available():
tools/get_flops.py:        model.cuda()
tools/benchmark.py:    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
tools/benchmark.py:    if samples_per_gpu > 1:
tools/benchmark.py:        samples_per_gpu=1,
tools/benchmark.py:        workers_per_gpu=cfg.data.workers_per_gpu,
tools/benchmark.py:        torch.cuda.synchronize()
tools/benchmark.py:        torch.cuda.synchronize()
tools/test.py:from mmdet.apis import multi_gpu_test, single_gpu_test
tools/test.py:        '--gpu-collect',
tools/test.py:        help='whether to use gpu to collect results.')
tools/test.py:        'workers, available when gpu-collect is not specified')
tools/test.py:    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
tools/test.py:    if samples_per_gpu > 1:
tools/test.py:        samples_per_gpu=samples_per_gpu,
tools/test.py:        workers_per_gpu=cfg.data.workers_per_gpu,
tools/test.py:        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
tools/test.py:            model.cuda(),
tools/test.py:            device_ids=[torch.cuda.current_device()],
tools/test.py:        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
tools/test.py:                                 args.gpu_collect)
tools/test.py:                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
tools/test_robustness.py:from mmdet.apis import multi_gpu_test, set_random_seed, single_gpu_test
tools/test_robustness.py:        '--workers', type=int, default=32, help='workers per gpu')
tools/test_robustness.py:        args.workers = cfg.data.workers_per_gpu
tools/test_robustness.py:            # TODO: support multiple images per gpu
tools/test_robustness.py:                samples_per_gpu=1,
tools/test_robustness.py:                workers_per_gpu=args.workers,
tools/test_robustness.py:                outputs = single_gpu_test(model, data_loader, args.show,
tools/test_robustness.py:                    model.cuda(),
tools/test_robustness.py:                    device_ids=[torch.cuda.current_device()],
tools/test_robustness.py:                outputs = multi_gpu_test(model, data_loader, args.tmpdir)
tools/train.py:    group_gpus = parser.add_mutually_exclusive_group()
tools/train.py:    group_gpus.add_argument(
tools/train.py:        '--gpus',
tools/train.py:        help='number of gpus to use '
tools/train.py:    group_gpus.add_argument(
tools/train.py:        '--gpu-ids',
tools/train.py:        help='ids of gpus to use '
tools/train.py:    if args.gpu_ids is not None:
tools/train.py:        cfg.gpu_ids = args.gpu_ids
tools/train.py:        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
tools/train.py:        # re-set gpu_ids with distributed training mode
tools/train.py:        cfg.gpu_ids = range(world_size)
tools/slurm_test.sh:GPUS=${GPUS:-8}
tools/slurm_test.sh:GPUS_PER_NODE=${GPUS_PER_NODE:-8}
tools/slurm_test.sh:    --gres=gpu:${GPUS_PER_NODE} \
tools/slurm_test.sh:    --ntasks=${GPUS} \
tools/slurm_test.sh:    --ntasks-per-node=${GPUS_PER_NODE} \
tools/dist_test.sh:GPUS=$3
tools/dist_test.sh:python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
tools/eval_metric.py:                'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
tools/dist_train.sh:GPUS=$2
tools/dist_train.sh:python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
build/lib/mmdet/datasets/samplers/group_sampler.py:    def __init__(self, dataset, samples_per_gpu=1):
build/lib/mmdet/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
build/lib/mmdet/datasets/samplers/group_sampler.py:                size / self.samples_per_gpu)) * self.samples_per_gpu
build/lib/mmdet/datasets/samplers/group_sampler.py:            num_extra = int(np.ceil(size / self.samples_per_gpu)
build/lib/mmdet/datasets/samplers/group_sampler.py:                            ) * self.samples_per_gpu - len(indice)
build/lib/mmdet/datasets/samplers/group_sampler.py:            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
build/lib/mmdet/datasets/samplers/group_sampler.py:                range(len(indices) // self.samples_per_gpu))
build/lib/mmdet/datasets/samplers/group_sampler.py:                 samples_per_gpu=1,
build/lib/mmdet/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
build/lib/mmdet/datasets/samplers/group_sampler.py:                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
build/lib/mmdet/datasets/samplers/group_sampler.py:                          self.num_replicas)) * self.samples_per_gpu
build/lib/mmdet/datasets/samplers/group_sampler.py:                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
build/lib/mmdet/datasets/samplers/group_sampler.py:                ) * self.samples_per_gpu * self.num_replicas - len(indice)
build/lib/mmdet/datasets/samplers/group_sampler.py:                    len(indices) // self.samples_per_gpu, generator=g))
build/lib/mmdet/datasets/samplers/group_sampler.py:            for j in range(i * self.samples_per_gpu, (i + 1) *
build/lib/mmdet/datasets/samplers/group_sampler.py:                           self.samples_per_gpu)
build/lib/mmdet/datasets/builder.py:                     samples_per_gpu,
build/lib/mmdet/datasets/builder.py:                     workers_per_gpu,
build/lib/mmdet/datasets/builder.py:                     num_gpus=1,
build/lib/mmdet/datasets/builder.py:    In distributed training, each GPU/process has a dataloader.
build/lib/mmdet/datasets/builder.py:    In non-distributed training, there is only one dataloader for all GPUs.
build/lib/mmdet/datasets/builder.py:        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
build/lib/mmdet/datasets/builder.py:            batch size of each GPU.
build/lib/mmdet/datasets/builder.py:        workers_per_gpu (int): How many subprocesses to use for data loading
build/lib/mmdet/datasets/builder.py:            for each GPU.
build/lib/mmdet/datasets/builder.py:        num_gpus (int): Number of GPUs. Only used in non-distributed training.
build/lib/mmdet/datasets/builder.py:        # that images on each GPU are in the same group
build/lib/mmdet/datasets/builder.py:            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
build/lib/mmdet/datasets/builder.py:        batch_size = samples_per_gpu
build/lib/mmdet/datasets/builder.py:        num_workers = workers_per_gpu
build/lib/mmdet/datasets/builder.py:        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
build/lib/mmdet/datasets/builder.py:        batch_size = num_gpus * samples_per_gpu
build/lib/mmdet/datasets/builder.py:        num_workers = num_gpus * workers_per_gpu
build/lib/mmdet/datasets/builder.py:        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
build/lib/mmdet/models/detectors/base.py:        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
build/lib/mmdet/models/detectors/base.py:        samples_per_gpu = img[0].size(0)
build/lib/mmdet/models/detectors/base.py:        assert samples_per_gpu == 1
build/lib/mmdet/models/detectors/base.py:                DDP, it means the batch size on each GPU), which is used for \
build/lib/mmdet/models/losses/focal_loss.py:    r"""A warpper of cuda version `Focal Loss
build/lib/mmdet/models/dense_heads/anchor_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/models/dense_heads/vfnet_head.py:            examples across GPUs. Default: True
build/lib/mmdet/models/dense_heads/vfnet_head.py:        # sync num_pos across all gpus
build/lib/mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = reduce_mean(
build/lib/mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
build/lib/mmdet/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = num_pos
build/lib/mmdet/models/dense_heads/vfnet_head.py:            iou_targets_ini_avg_per_gpu = reduce_mean(
build/lib/mmdet/models/dense_heads/vfnet_head.py:            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
build/lib/mmdet/models/dense_heads/vfnet_head.py:            iou_targets_rf_avg_per_gpu = reduce_mean(
build/lib/mmdet/models/dense_heads/vfnet_head.py:            bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)
build/lib/mmdet/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
build/lib/mmdet/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
build/lib/mmdet/models/dense_heads/transformer_head.py:        # Compute the average number of gt boxes accross all gpus, for
build/lib/mmdet/models/dense_heads/sabl_retina_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/models/dense_heads/guided_anchor_head.py:    def get_sampled_approxs(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/models/dense_heads/guided_anchor_head.py:                    device='cuda'):
build/lib/mmdet/models/dense_heads/guided_anchor_head.py:        img_per_gpu = len(gt_bboxes_list)
build/lib/mmdet/models/dense_heads/guided_anchor_head.py:                img_per_gpu,
build/lib/mmdet/models/dense_heads/guided_anchor_head.py:        for img_id in range(img_per_gpu):
build/lib/mmdet/models/dense_heads/gfl_head.py:                reduced over all GPUs.
build/lib/mmdet/models/dense_heads/gfl_head.py:            weight_targets = torch.tensor(0).cuda()
build/lib/mmdet/models/dense_heads/atss_head.py:                reduced over all GPUs.
build/lib/mmdet/models/dense_heads/atss_head.py:            torch.tensor(num_total_pos).cuda()).item()
build/lib/mmdet/models/dense_heads/rpn_test_mixin.py:        samples_per_gpu = len(img_metas[0])
build/lib/mmdet/models/dense_heads/rpn_test_mixin.py:        aug_proposals = [[] for _ in range(samples_per_gpu)]
build/lib/mmdet/models/dense_heads/rpn_test_mixin.py:        for i in range(samples_per_gpu):
build/lib/mmdet/models/roi_heads/bbox_heads/bbox_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
build/lib/mmdet/models/roi_heads/bbox_heads/sabl_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
build/lib/mmdet/models/roi_heads/mask_heads/d2det_head.py:        targets = targets.cuda()
build/lib/mmdet/models/roi_heads/mask_heads/d2det_head.py:        points = points.cuda()
build/lib/mmdet/models/roi_heads/mask_heads/d2det_head.py:        masks = masks.cuda()
build/lib/mmdet/models/roi_heads/mask_heads/d2det_head.py:        idx = (torch.arange(0, map_size).float() + 0.5).cuda() / map_size
build/lib/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
build/lib/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:            # GPU benefits from parallelism for larger chunks,
build/lib/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
build/lib/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
build/lib/mmdet/models/roi_heads/mask_heads/fcn_mask_head.py:    # On GPU, paste all masks together (up to chunk size)
build/lib/mmdet/core/evaluation/eval_hooks.py:        from mmdet.apis import single_gpu_test
build/lib/mmdet/core/evaluation/eval_hooks.py:        results = single_gpu_test(runner.model, self.dataloader, show=False)
build/lib/mmdet/core/evaluation/eval_hooks.py:        gpu_collect (bool): Whether to use gpu or cpu to collect results.
build/lib/mmdet/core/evaluation/eval_hooks.py:                 gpu_collect=False,
build/lib/mmdet/core/evaluation/eval_hooks.py:        self.gpu_collect = gpu_collect
build/lib/mmdet/core/evaluation/eval_hooks.py:        from mmdet.apis import multi_gpu_test
build/lib/mmdet/core/evaluation/eval_hooks.py:        results = multi_gpu_test(
build/lib/mmdet/core/evaluation/eval_hooks.py:            gpu_collect=self.gpu_collect)
build/lib/mmdet/core/post_processing/bbox_nms.py:    implement Fast NMS entirely in standard GPU-accelerated matrix operations.
build/lib/mmdet/core/anchor/anchor_generator.py:    def grid_anchors(self, featmap_sizes, device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                                  device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/core/anchor/anchor_generator.py:    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                                 device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/core/anchor/anchor_generator.py:    def responsible_flags(self, featmap_sizes, gt_bboxes, device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                                       device='cuda'):
build/lib/mmdet/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/core/anchor/point_generator.py:    def grid_points(self, featmap_size, stride=16, device='cuda'):
build/lib/mmdet/core/anchor/point_generator.py:    def valid_flags(self, featmap_size, valid_size, device='cuda'):
build/lib/mmdet/core/bbox/assigners/max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
build/lib/mmdet/core/bbox/assigners/max_iou_assigner.py:                 gpu_assign_thr=-1,
build/lib/mmdet/core/bbox/assigners/max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
build/lib/mmdet/core/bbox/assigners/max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
build/lib/mmdet/core/bbox/assigners/max_iou_assigner.py:            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
build/lib/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
build/lib/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:                 gpu_assign_thr=-1,
build/lib/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
build/lib/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
build/lib/mmdet/core/bbox/assigners/approx_max_iou_assigner.py:            num_gts > self.gpu_assign_thr) else False
build/lib/mmdet/core/bbox/samplers/random_sampler.py:            if torch.cuda.is_available():
build/lib/mmdet/core/bbox/samplers/random_sampler.py:                device = torch.cuda.current_device()
build/lib/mmdet/core/bbox/samplers/sampling_result.py:            >>> # xdoctest: +REQUIRES(--gpu)
build/lib/mmdet/core/bbox/samplers/score_hlr_sampler.py:            if torch.cuda.is_available():
build/lib/mmdet/core/bbox/samplers/score_hlr_sampler.py:                device = torch.cuda.current_device()
build/lib/mmdet/core/utils/dist_utils.py:    """"Obtain the mean of tensor on different GPUs."""
build/lib/mmdet/old/datasets/samplers/group_sampler.py:    def __init__(self, dataset, samples_per_gpu=1):
build/lib/mmdet/old/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                size / self.samples_per_gpu)) * self.samples_per_gpu
build/lib/mmdet/old/datasets/samplers/group_sampler.py:            num_extra = int(np.ceil(size / self.samples_per_gpu)
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                            ) * self.samples_per_gpu - len(indice)
build/lib/mmdet/old/datasets/samplers/group_sampler.py:            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                range(len(indices) // self.samples_per_gpu))
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                 samples_per_gpu=1,
build/lib/mmdet/old/datasets/samplers/group_sampler.py:        self.samples_per_gpu = samples_per_gpu
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                          self.num_replicas)) * self.samples_per_gpu
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                ) * self.samples_per_gpu * self.num_replicas - len(indice)
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                    len(indices) // self.samples_per_gpu, generator=g))
build/lib/mmdet/old/datasets/samplers/group_sampler.py:            for j in range(i * self.samples_per_gpu, (i + 1) *
build/lib/mmdet/old/datasets/samplers/group_sampler.py:                           self.samples_per_gpu)
build/lib/mmdet/old/datasets/builder.py:                     samples_per_gpu,
build/lib/mmdet/old/datasets/builder.py:                     workers_per_gpu,
build/lib/mmdet/old/datasets/builder.py:                     num_gpus=1,
build/lib/mmdet/old/datasets/builder.py:    In distributed training, each GPU/process has a dataloader.
build/lib/mmdet/old/datasets/builder.py:    In non-distributed training, there is only one dataloader for all GPUs.
build/lib/mmdet/old/datasets/builder.py:        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
build/lib/mmdet/old/datasets/builder.py:            batch size of each GPU.
build/lib/mmdet/old/datasets/builder.py:        workers_per_gpu (int): How many subprocesses to use for data loading
build/lib/mmdet/old/datasets/builder.py:            for each GPU.
build/lib/mmdet/old/datasets/builder.py:        num_gpus (int): Number of GPUs. Only used in non-distributed training.
build/lib/mmdet/old/datasets/builder.py:        # that images on each GPU are in the same group
build/lib/mmdet/old/datasets/builder.py:            sampler = DistributedGroupSampler(dataset, samples_per_gpu,
build/lib/mmdet/old/datasets/builder.py:        batch_size = samples_per_gpu
build/lib/mmdet/old/datasets/builder.py:        num_workers = workers_per_gpu
build/lib/mmdet/old/datasets/builder.py:        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
build/lib/mmdet/old/datasets/builder.py:        batch_size = num_gpus * samples_per_gpu
build/lib/mmdet/old/datasets/builder.py:        num_workers = num_gpus * workers_per_gpu
build/lib/mmdet/old/datasets/builder.py:        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
build/lib/mmdet/old/models/detectors/base.py:        # TODO: remove the restriction of samples_per_gpu == 1 when prepared
build/lib/mmdet/old/models/detectors/base.py:        samples_per_gpu = img[0].size(0)
build/lib/mmdet/old/models/detectors/base.py:        assert samples_per_gpu == 1
build/lib/mmdet/old/models/detectors/base.py:                DDP, it means the batch size on each GPU), which is used for \
build/lib/mmdet/old/models/losses/focal_loss.py:    r"""A warpper of cuda version `Focal Loss
build/lib/mmdet/old/models/dense_heads/anchor_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            examples across GPUs. Default: True
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:        # sync num_pos across all gpus
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = reduce_mean(
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            num_pos_avg_per_gpu = num_pos
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            iou_targets_ini_avg_per_gpu = reduce_mean(
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            bbox_avg_factor_ini = max(iou_targets_ini_avg_per_gpu, 1.0)
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            iou_targets_rf_avg_per_gpu = reduce_mean(
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:            bbox_avg_factor_rf = max(iou_targets_rf_avg_per_gpu, 1.0)
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
build/lib/mmdet/old/models/dense_heads/vfnet_head.py:                avg_factor=num_pos_avg_per_gpu)
build/lib/mmdet/old/models/dense_heads/transformer_head.py:        # Compute the average number of gt boxes accross all gpus, for
build/lib/mmdet/old/models/dense_heads/sabl_retina_head.py:    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/old/models/dense_heads/guided_anchor_head.py:    def get_sampled_approxs(self, featmap_sizes, img_metas, device='cuda'):
build/lib/mmdet/old/models/dense_heads/guided_anchor_head.py:                    device='cuda'):
build/lib/mmdet/old/models/dense_heads/guided_anchor_head.py:        img_per_gpu = len(gt_bboxes_list)
build/lib/mmdet/old/models/dense_heads/guided_anchor_head.py:                img_per_gpu,
build/lib/mmdet/old/models/dense_heads/guided_anchor_head.py:        for img_id in range(img_per_gpu):
build/lib/mmdet/old/models/dense_heads/gfl_head.py:                reduced over all GPUs.
build/lib/mmdet/old/models/dense_heads/gfl_head.py:            weight_targets = torch.tensor(0).cuda()
build/lib/mmdet/old/models/dense_heads/atss_head.py:                reduced over all GPUs.
build/lib/mmdet/old/models/dense_heads/atss_head.py:            torch.tensor(num_total_pos).cuda()).item()
build/lib/mmdet/old/models/dense_heads/rpn_test_mixin.py:        samples_per_gpu = len(img_metas[0])
build/lib/mmdet/old/models/dense_heads/rpn_test_mixin.py:        aug_proposals = [[] for _ in range(samples_per_gpu)]
build/lib/mmdet/old/models/dense_heads/rpn_test_mixin.py:        for i in range(samples_per_gpu):
build/lib/mmdet/old/models/roi_heads/bbox_heads/bbox_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
build/lib/mmdet/old/models/roi_heads/bbox_heads/sabl_head.py:            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
build/lib/mmdet/old/models/roi_heads/mask_heads/d2det_head.py:        targets = targets.cuda()
build/lib/mmdet/old/models/roi_heads/mask_heads/d2det_head.py:        points = points.cuda()
build/lib/mmdet/old/models/roi_heads/mask_heads/d2det_head.py:        masks = masks.cuda()
build/lib/mmdet/old/models/roi_heads/mask_heads/d2det_head.py:        idx = (torch.arange(0, map_size).float() + 0.5).cuda() / map_size
build/lib/mmdet/old/models/roi_heads/mask_heads/fcn_mask_head.py:GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit
build/lib/mmdet/old/models/roi_heads/mask_heads/fcn_mask_head.py:            # GPU benefits from parallelism for larger chunks,
build/lib/mmdet/old/models/roi_heads/mask_heads/fcn_mask_head.py:                np.ceil(N * img_h * img_w * BYTES_PER_FLOAT / GPU_MEM_LIMIT))
build/lib/mmdet/old/models/roi_heads/mask_heads/fcn_mask_head.py:                    N), 'Default GPU_MEM_LIMIT is too small; try increasing it'
build/lib/mmdet/old/models/roi_heads/mask_heads/fcn_mask_head.py:    # On GPU, paste all masks together (up to chunk size)
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        from mmdet.apis import single_gpu_test
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        results = single_gpu_test(runner.model, self.dataloader, show=False)
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        gpu_collect (bool): Whether to use gpu or cpu to collect results.
build/lib/mmdet/old/core/evaluation/eval_hooks.py:                 gpu_collect=False,
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        self.gpu_collect = gpu_collect
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        from mmdet.apis import multi_gpu_test
build/lib/mmdet/old/core/evaluation/eval_hooks.py:        results = multi_gpu_test(
build/lib/mmdet/old/core/evaluation/eval_hooks.py:            gpu_collect=self.gpu_collect)
build/lib/mmdet/old/core/post_processing/bbox_nms.py:    implement Fast NMS entirely in standard GPU-accelerated matrix operations.
build/lib/mmdet/old/core/anchor/anchor_generator.py:    def grid_anchors(self, featmap_sizes, device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                                  device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/old/core/anchor/anchor_generator.py:    def valid_flags(self, featmap_sizes, pad_shape, device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                                 device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/old/core/anchor/anchor_generator.py:    def responsible_flags(self, featmap_sizes, gt_bboxes, device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                                       device='cuda'):
build/lib/mmdet/old/core/anchor/anchor_generator.py:                Defaults to 'cuda'.
build/lib/mmdet/old/core/anchor/point_generator.py:    def grid_points(self, featmap_size, stride=16, device='cuda'):
build/lib/mmdet/old/core/anchor/point_generator.py:    def valid_flags(self, featmap_size, valid_size, device='cuda'):
build/lib/mmdet/old/core/bbox/assigners/max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
build/lib/mmdet/old/core/bbox/assigners/max_iou_assigner.py:                 gpu_assign_thr=-1,
build/lib/mmdet/old/core/bbox/assigners/max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
build/lib/mmdet/old/core/bbox/assigners/max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
build/lib/mmdet/old/core/bbox/assigners/max_iou_assigner.py:            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
build/lib/mmdet/old/core/bbox/assigners/approx_max_iou_assigner.py:        gpu_assign_thr (int): The upper bound of the number of GT for GPU
build/lib/mmdet/old/core/bbox/assigners/approx_max_iou_assigner.py:                 gpu_assign_thr=-1,
build/lib/mmdet/old/core/bbox/assigners/approx_max_iou_assigner.py:        self.gpu_assign_thr = gpu_assign_thr
build/lib/mmdet/old/core/bbox/assigners/approx_max_iou_assigner.py:        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
build/lib/mmdet/old/core/bbox/assigners/approx_max_iou_assigner.py:            num_gts > self.gpu_assign_thr) else False
build/lib/mmdet/old/core/bbox/samplers/random_sampler.py:            if torch.cuda.is_available():
build/lib/mmdet/old/core/bbox/samplers/random_sampler.py:                device = torch.cuda.current_device()
build/lib/mmdet/old/core/bbox/samplers/sampling_result.py:            >>> # xdoctest: +REQUIRES(--gpu)
build/lib/mmdet/old/core/bbox/samplers/score_hlr_sampler.py:            if torch.cuda.is_available():
build/lib/mmdet/old/core/bbox/samplers/score_hlr_sampler.py:                device = torch.cuda.current_device()
build/lib/mmdet/old/core/utils/dist_utils.py:    """"Obtain the mean of tensor on different GPUs."""
build/lib/mmdet/old/ops/__init__.py:                      get_compiling_cuda_version, modulated_deform_conv, nms,
build/lib/mmdet/old/ops/__init__.py:    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
build/lib/mmdet/old/utils/contextmanagers.py:                    streams: List[torch.cuda.Stream] = None):
build/lib/mmdet/old/utils/contextmanagers.py:    """Async context manager that waits for work to complete on given CUDA
build/lib/mmdet/old/utils/contextmanagers.py:    if not torch.cuda.is_available():
build/lib/mmdet/old/utils/contextmanagers.py:    stream_before_context_switch = torch.cuda.current_stream()
build/lib/mmdet/old/utils/contextmanagers.py:        torch.cuda.Event(enable_timing=DEBUG_COMPLETED_TIME) for _ in streams
build/lib/mmdet/old/utils/contextmanagers.py:        start = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/old/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
build/lib/mmdet/old/utils/contextmanagers.py:        with torch.cuda.stream(stream_before_context_switch):
build/lib/mmdet/old/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
build/lib/mmdet/old/utils/contextmanagers.py:    if not torch.cuda.is_available():
build/lib/mmdet/old/utils/contextmanagers.py:    initial_stream = torch.cuda.current_stream()
build/lib/mmdet/old/utils/contextmanagers.py:    with torch.cuda.stream(initial_stream):
build/lib/mmdet/old/utils/contextmanagers.py:        assert isinstance(stream, torch.cuda.Stream)
build/lib/mmdet/old/utils/contextmanagers.py:            with torch.cuda.stream(stream):
build/lib/mmdet/old/utils/contextmanagers.py:                current = torch.cuda.current_stream()
build/lib/mmdet/old/utils/profiling.py:        """Print time spent by CPU and GPU.
build/lib/mmdet/old/utils/profiling.py:        if (not enabled) or not torch.cuda.is_available():
build/lib/mmdet/old/utils/profiling.py:        stream = stream if stream else torch.cuda.current_stream()
build/lib/mmdet/old/utils/profiling.py:        start = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/old/utils/profiling.py:        end = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/old/utils/profiling.py:            gpu_time = start.elapsed_time(end)
build/lib/mmdet/old/utils/profiling.py:            msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
build/lib/mmdet/old/apis/test.py:def single_gpu_test(model,
build/lib/mmdet/old/apis/test.py:def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
build/lib/mmdet/old/apis/test.py:    """Test model with multiple gpus.
build/lib/mmdet/old/apis/test.py:    This method tests model with multiple gpus and collects the results
build/lib/mmdet/old/apis/test.py:    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
build/lib/mmdet/old/apis/test.py:    it encodes results to gpu tensors and use gpu communication for results
build/lib/mmdet/old/apis/test.py:    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
build/lib/mmdet/old/apis/test.py:            different gpus under cpu mode.
build/lib/mmdet/old/apis/test.py:        gpu_collect (bool): Option to use either gpu or cpu to collect results.
build/lib/mmdet/old/apis/test.py:    if gpu_collect:
build/lib/mmdet/old/apis/test.py:        results = collect_results_gpu(results, len(dataset))
build/lib/mmdet/old/apis/test.py:                                device='cuda')
build/lib/mmdet/old/apis/test.py:                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
build/lib/mmdet/old/apis/test.py:def collect_results_gpu(result_part, size):
build/lib/mmdet/old/apis/test.py:        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
build/lib/mmdet/old/apis/test.py:    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
build/lib/mmdet/old/apis/test.py:    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
build/lib/mmdet/old/apis/inference.py:def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
build/lib/mmdet/old/apis/inference.py:    data = collate([data], samples_per_gpu=1)
build/lib/mmdet/old/apis/inference.py:    if next(model.parameters()).is_cuda:
build/lib/mmdet/old/apis/inference.py:        # scatter to specified GPU
build/lib/mmdet/old/apis/inference.py:    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
build/lib/mmdet/old/apis/train.py:    torch.cuda.manual_seed_all(seed)
build/lib/mmdet/old/apis/train.py:    if 'imgs_per_gpu' in cfg.data:
build/lib/mmdet/old/apis/train.py:        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
build/lib/mmdet/old/apis/train.py:                       'Please use "samples_per_gpu" instead')
build/lib/mmdet/old/apis/train.py:        if 'samples_per_gpu' in cfg.data:
build/lib/mmdet/old/apis/train.py:                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
build/lib/mmdet/old/apis/train.py:                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
build/lib/mmdet/old/apis/train.py:                f'={cfg.data.imgs_per_gpu} is used in this experiments')
build/lib/mmdet/old/apis/train.py:                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
build/lib/mmdet/old/apis/train.py:                f'{cfg.data.imgs_per_gpu} in this experiments')
build/lib/mmdet/old/apis/train.py:        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
build/lib/mmdet/old/apis/train.py:            cfg.data.samples_per_gpu,
build/lib/mmdet/old/apis/train.py:            cfg.data.workers_per_gpu,
build/lib/mmdet/old/apis/train.py:            # cfg.gpus will be ignored if distributed
build/lib/mmdet/old/apis/train.py:            len(cfg.gpu_ids),
build/lib/mmdet/old/apis/train.py:    # put model on gpus
build/lib/mmdet/old/apis/train.py:            model.cuda(),
build/lib/mmdet/old/apis/train.py:            device_ids=[torch.cuda.current_device()],
build/lib/mmdet/old/apis/train.py:            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
build/lib/mmdet/old/apis/train.py:        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
build/lib/mmdet/old/apis/train.py:        if val_samples_per_gpu > 1:
build/lib/mmdet/old/apis/train.py:            samples_per_gpu=val_samples_per_gpu,
build/lib/mmdet/old/apis/train.py:            workers_per_gpu=cfg.data.workers_per_gpu,
build/lib/mmdet/old/apis/__init__.py:from .test import multi_gpu_test, single_gpu_test
build/lib/mmdet/old/apis/__init__.py:    'multi_gpu_test', 'single_gpu_test'
build/lib/mmdet/ops/dcn/deform_pool.py:        if not data.is_cuda:
build/lib/mmdet/ops/dcn/deform_pool.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops/dcn/deform_conv.py:        if not input.is_cuda:
build/lib/mmdet/ops/dcn/deform_conv.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops/dcn/deform_conv.py:        if not input.is_cuda:
build/lib/mmdet/ops/dcn/deform_conv.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops/dcn/deform_conv.py:        # To fix an assert error in deform_conv_cuda.cpp:128
build/lib/mmdet/ops/masked_conv/masked_conv.py:        if not features.is_cuda:
build/lib/mmdet/ops/roi_align/roi_align.py:        elif features.is_cuda:
build/lib/mmdet/ops/roi_align/gradcheck.py:    num_imgs, 16, feat_size, feat_size, requires_grad=True, device='cuda:0')
build/lib/mmdet/ops/roi_align/gradcheck.py:rois = torch.from_numpy(rois).float().cuda()
build/lib/mmdet/ops/roi_pool/roi_pool.py:        assert features.is_cuda
build/lib/mmdet/ops/roi_pool/roi_pool.py:        assert grad_output.is_cuda
build/lib/mmdet/ops/roi_pool/gradcheck.py:feat = torch.randn(4, 16, 15, 15, requires_grad=True).cuda()
build/lib/mmdet/ops/roi_pool/gradcheck.py:                     [1, 67, 40, 110, 120]]).cuda()
build/lib/mmdet/ops/generalized_attention.py:        h_idxs = torch.linspace(0, h - 1, h).cuda(device)
build/lib/mmdet/ops/generalized_attention.py:        w_idxs = torch.linspace(0, w - 1, w).cuda(device)
build/lib/mmdet/ops/generalized_attention.py:        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).cuda(device)
build/lib/mmdet/ops/generalized_attention.py:        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).cuda(device)
build/lib/mmdet/ops/generalized_attention.py:        feat_range = torch.arange(0, feat_dim / 4).cuda(device)
build/lib/mmdet/ops/generalized_attention.py:        dim_mat = torch.Tensor([wave_length]).cuda(device)
build/lib/mmdet/ops/carafe/setup.py:from torch.utils.cpp_extension import BuildExtension, CUDAExtension
build/lib/mmdet/ops/carafe/setup.py:    '-D__CUDA_NO_HALF_OPERATORS__',
build/lib/mmdet/ops/carafe/setup.py:    '-D__CUDA_NO_HALF_CONVERSIONS__',
build/lib/mmdet/ops/carafe/setup.py:    '-D__CUDA_NO_HALF2_OPERATORS__',
build/lib/mmdet/ops/carafe/setup.py:        CUDAExtension(
build/lib/mmdet/ops/carafe/setup.py:                'src/cuda/carafe_cuda.cpp', 'src/cuda/carafe_cuda_kernel.cu',
build/lib/mmdet/ops/carafe/setup.py:            define_macros=[('WITH_CUDA', None)],
build/lib/mmdet/ops/carafe/setup.py:        CUDAExtension(
build/lib/mmdet/ops/carafe/setup.py:                'src/cuda/carafe_naive_cuda.cpp',
build/lib/mmdet/ops/carafe/setup.py:                'src/cuda/carafe_naive_cuda_kernel.cu',
build/lib/mmdet/ops/carafe/setup.py:            define_macros=[('WITH_CUDA', None)],
build/lib/mmdet/ops/carafe/carafe.py:        if features.is_cuda:
build/lib/mmdet/ops/carafe/carafe.py:        assert grad_output.is_cuda
build/lib/mmdet/ops/carafe/carafe.py:        if features.is_cuda:
build/lib/mmdet/ops/carafe/carafe.py:        assert grad_output.is_cuda
build/lib/mmdet/ops/carafe/grad_check.py:feat = torch.randn(2, 64, 3, 3, requires_grad=True, device='cuda:0').double()
build/lib/mmdet/ops/carafe/grad_check.py:    2, 100, 6, 6, requires_grad=True, device='cuda:0').sigmoid().double()
build/lib/mmdet/ops/carafe/grad_check.py:    2, 1024, 100, 100, requires_grad=True, device='cuda:0').float()
build/lib/mmdet/ops/carafe/grad_check.py:    2, 25, 200, 200, requires_grad=True, device='cuda:0').sigmoid().float()
build/lib/mmdet/ops/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops/sigmoid_focal_loss/sigmoid_focal_loss.py:        assert logits.is_cuda
build/lib/mmdet/ops/nms/nms_wrapper.py:    """Dispatch to either CPU or GPU NMS implementations.
build/lib/mmdet/ops/nms/nms_wrapper.py:    The input can be either a torch tensor or numpy array. GPU NMS will be used
build/lib/mmdet/ops/nms/nms_wrapper.py:    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
build/lib/mmdet/ops/nms/nms_wrapper.py:            is None, then cpu nms is used, otherwise gpu_nms will be used.
build/lib/mmdet/ops/nms/nms_wrapper.py:        device = 'cpu' if device_id is None else f'cuda:{device_id}'
build/lib/mmdet/ops/nms/nms_wrapper.py:    # execute cpu or cuda nms
build/lib/mmdet/ops/nms/nms_wrapper.py:        if dets_th.is_cuda:
build/lib/mmdet/ops/utils/__init__.py:from .compiling_info import get_compiler_version, get_compiling_cuda_version
build/lib/mmdet/ops/utils/__init__.py:# get_compiling_cuda_version = compiling_info.get_compiling_cuda_version
build/lib/mmdet/ops/utils/__init__.py:__all__ = ['get_compiler_version', 'get_compiling_cuda_version']
build/lib/mmdet/ops/__init__.py:                      get_compiling_cuda_version, modulated_deform_conv, nms,
build/lib/mmdet/ops/__init__.py:    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
build/lib/mmdet/ops2/dcn/deform_pool.py:        if not data.is_cuda:
build/lib/mmdet/ops2/dcn/deform_pool.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops2/dcn/deform_conv.py:        if not input.is_cuda:
build/lib/mmdet/ops2/dcn/deform_conv.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops2/dcn/deform_conv.py:        if not input.is_cuda:
build/lib/mmdet/ops2/dcn/deform_conv.py:        if not grad_output.is_cuda:
build/lib/mmdet/ops2/dcn/deform_conv.py:        # To fix an assert error in deform_conv_cuda.cpp:128
build/lib/mmdet/ops2/masked_conv/masked_conv.py:        if not features.is_cuda:
build/lib/mmdet/ops2/roi_align/roi_align.py:        elif features.is_cuda:
build/lib/mmdet/ops2/roi_align/gradcheck.py:    num_imgs, 16, feat_size, feat_size, requires_grad=True, device='cuda:0')
build/lib/mmdet/ops2/roi_align/gradcheck.py:rois = torch.from_numpy(rois).float().cuda()
build/lib/mmdet/ops2/roi_pool/roi_pool.py:        assert features.is_cuda
build/lib/mmdet/ops2/roi_pool/roi_pool.py:        assert grad_output.is_cuda
build/lib/mmdet/ops2/roi_pool/gradcheck.py:feat = torch.randn(4, 16, 15, 15, requires_grad=True).cuda()
build/lib/mmdet/ops2/roi_pool/gradcheck.py:                     [1, 67, 40, 110, 120]]).cuda()
build/lib/mmdet/ops2/generalized_attention.py:        h_idxs = torch.linspace(0, h - 1, h).cuda(device)
build/lib/mmdet/ops2/generalized_attention.py:        w_idxs = torch.linspace(0, w - 1, w).cuda(device)
build/lib/mmdet/ops2/generalized_attention.py:        h_kv_idxs = torch.linspace(0, h_kv - 1, h_kv).cuda(device)
build/lib/mmdet/ops2/generalized_attention.py:        w_kv_idxs = torch.linspace(0, w_kv - 1, w_kv).cuda(device)
build/lib/mmdet/ops2/generalized_attention.py:        feat_range = torch.arange(0, feat_dim / 4).cuda(device)
build/lib/mmdet/ops2/generalized_attention.py:        dim_mat = torch.Tensor([wave_length]).cuda(device)
build/lib/mmdet/ops2/carafe/setup.py:from torch.utils.cpp_extension import BuildExtension, CUDAExtension
build/lib/mmdet/ops2/carafe/setup.py:    '-D__CUDA_NO_HALF_OPERATORS__',
build/lib/mmdet/ops2/carafe/setup.py:    '-D__CUDA_NO_HALF_CONVERSIONS__',
build/lib/mmdet/ops2/carafe/setup.py:    '-D__CUDA_NO_HALF2_OPERATORS__',
build/lib/mmdet/ops2/carafe/setup.py:        CUDAExtension(
build/lib/mmdet/ops2/carafe/setup.py:                'src/cuda/carafe_cuda.cpp', 'src/cuda/carafe_cuda_kernel.cu',
build/lib/mmdet/ops2/carafe/setup.py:            define_macros=[('WITH_CUDA', None)],
build/lib/mmdet/ops2/carafe/setup.py:        CUDAExtension(
build/lib/mmdet/ops2/carafe/setup.py:                'src/cuda/carafe_naive_cuda.cpp',
build/lib/mmdet/ops2/carafe/setup.py:                'src/cuda/carafe_naive_cuda_kernel.cu',
build/lib/mmdet/ops2/carafe/setup.py:            define_macros=[('WITH_CUDA', None)],
build/lib/mmdet/ops2/carafe/carafe.py:        if features.is_cuda:
build/lib/mmdet/ops2/carafe/carafe.py:        assert grad_output.is_cuda
build/lib/mmdet/ops2/carafe/carafe.py:        if features.is_cuda:
build/lib/mmdet/ops2/carafe/carafe.py:        assert grad_output.is_cuda
build/lib/mmdet/ops2/carafe/grad_check.py:feat = torch.randn(2, 64, 3, 3, requires_grad=True, device='cuda:0').double()
build/lib/mmdet/ops2/carafe/grad_check.py:    2, 100, 6, 6, requires_grad=True, device='cuda:0').sigmoid().double()
build/lib/mmdet/ops2/carafe/grad_check.py:    2, 1024, 100, 100, requires_grad=True, device='cuda:0').float()
build/lib/mmdet/ops2/carafe/grad_check.py:    2, 25, 200, 200, requires_grad=True, device='cuda:0').sigmoid().float()
build/lib/mmdet/ops2/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops2/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops2/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops2/carafe/grad_check.py:    torch.cuda.synchronize()
build/lib/mmdet/ops2/sigmoid_focal_loss/sigmoid_focal_loss.py:        assert logits.is_cuda
build/lib/mmdet/ops2/nms/nms_wrapper.py:    """Dispatch to either CPU or GPU NMS implementations.
build/lib/mmdet/ops2/nms/nms_wrapper.py:    The input can be either a torch tensor or numpy array. GPU NMS will be used
build/lib/mmdet/ops2/nms/nms_wrapper.py:    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
build/lib/mmdet/ops2/nms/nms_wrapper.py:            is None, then cpu nms is used, otherwise gpu_nms will be used.
build/lib/mmdet/ops2/nms/nms_wrapper.py:        device = 'cpu' if device_id is None else f'cuda:{device_id}'
build/lib/mmdet/ops2/nms/nms_wrapper.py:    # execute cpu or cuda nms
build/lib/mmdet/ops2/nms/nms_wrapper.py:        if dets_th.is_cuda:
build/lib/mmdet/ops2/__init__o.py:                      get_compiling_cuda_version, modulated_deform_conv, nms,
build/lib/mmdet/ops2/__init__o.py:    'get_compiler_version', 'get_compiling_cuda_version', 'ConvWS2d',
build/lib/mmdet/ops2/utils/__init__.py:from .compiling_info import get_compiler_version, get_compiling_cuda_version
build/lib/mmdet/ops2/utils/__init__.py:# get_compiling_cuda_version = compiling_info.get_compiling_cuda_version
build/lib/mmdet/ops2/utils/__init__.py:__all__ = ['get_compiler_version', 'get_compiling_cuda_version']
build/lib/mmdet/ops2/__init__.py:from .utils import get_compiler_version, get_compiling_cuda_version
build/lib/mmdet/ops2/__init__.py:    'get_compiling_cuda_version',
build/lib/mmdet/utils/contextmanagers.py:                    streams: List[torch.cuda.Stream] = None):
build/lib/mmdet/utils/contextmanagers.py:    """Async context manager that waits for work to complete on given CUDA
build/lib/mmdet/utils/contextmanagers.py:    if not torch.cuda.is_available():
build/lib/mmdet/utils/contextmanagers.py:    stream_before_context_switch = torch.cuda.current_stream()
build/lib/mmdet/utils/contextmanagers.py:        torch.cuda.Event(enable_timing=DEBUG_COMPLETED_TIME) for _ in streams
build/lib/mmdet/utils/contextmanagers.py:        start = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
build/lib/mmdet/utils/contextmanagers.py:        with torch.cuda.stream(stream_before_context_switch):
build/lib/mmdet/utils/contextmanagers.py:        current_stream = torch.cuda.current_stream()
build/lib/mmdet/utils/contextmanagers.py:    if not torch.cuda.is_available():
build/lib/mmdet/utils/contextmanagers.py:    initial_stream = torch.cuda.current_stream()
build/lib/mmdet/utils/contextmanagers.py:    with torch.cuda.stream(initial_stream):
build/lib/mmdet/utils/contextmanagers.py:        assert isinstance(stream, torch.cuda.Stream)
build/lib/mmdet/utils/contextmanagers.py:            with torch.cuda.stream(stream):
build/lib/mmdet/utils/contextmanagers.py:                current = torch.cuda.current_stream()
build/lib/mmdet/utils/profiling.py:        """Print time spent by CPU and GPU.
build/lib/mmdet/utils/profiling.py:        if (not enabled) or not torch.cuda.is_available():
build/lib/mmdet/utils/profiling.py:        stream = stream if stream else torch.cuda.current_stream()
build/lib/mmdet/utils/profiling.py:        start = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/utils/profiling.py:        end = torch.cuda.Event(enable_timing=True)
build/lib/mmdet/utils/profiling.py:            gpu_time = start.elapsed_time(end)
build/lib/mmdet/utils/profiling.py:            msg += f'gpu_time {gpu_time:.2f} ms stream {stream}'
build/lib/mmdet/apis/test.py:def single_gpu_test(model,
build/lib/mmdet/apis/test.py:def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
build/lib/mmdet/apis/test.py:    """Test model with multiple gpus.
build/lib/mmdet/apis/test.py:    This method tests model with multiple gpus and collects the results
build/lib/mmdet/apis/test.py:    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
build/lib/mmdet/apis/test.py:    it encodes results to gpu tensors and use gpu communication for results
build/lib/mmdet/apis/test.py:    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
build/lib/mmdet/apis/test.py:            different gpus under cpu mode.
build/lib/mmdet/apis/test.py:        gpu_collect (bool): Option to use either gpu or cpu to collect results.
build/lib/mmdet/apis/test.py:    if gpu_collect:
build/lib/mmdet/apis/test.py:        results = collect_results_gpu(results, len(dataset))
build/lib/mmdet/apis/test.py:                                device='cuda')
build/lib/mmdet/apis/test.py:                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
build/lib/mmdet/apis/test.py:def collect_results_gpu(result_part, size):
build/lib/mmdet/apis/test.py:        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
build/lib/mmdet/apis/test.py:    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
build/lib/mmdet/apis/test.py:    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
build/lib/mmdet/apis/inference.py:def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
build/lib/mmdet/apis/inference.py:    data = collate([data], samples_per_gpu=1)
build/lib/mmdet/apis/inference.py:    if next(model.parameters()).is_cuda:
build/lib/mmdet/apis/inference.py:        # scatter to specified GPU
build/lib/mmdet/apis/inference.py:    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
build/lib/mmdet/apis/train.py:    torch.cuda.manual_seed_all(seed)
build/lib/mmdet/apis/train.py:    if 'imgs_per_gpu' in cfg.data:
build/lib/mmdet/apis/train.py:        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
build/lib/mmdet/apis/train.py:                       'Please use "samples_per_gpu" instead')
build/lib/mmdet/apis/train.py:        if 'samples_per_gpu' in cfg.data:
build/lib/mmdet/apis/train.py:                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
build/lib/mmdet/apis/train.py:                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
build/lib/mmdet/apis/train.py:                f'={cfg.data.imgs_per_gpu} is used in this experiments')
build/lib/mmdet/apis/train.py:                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
build/lib/mmdet/apis/train.py:                f'{cfg.data.imgs_per_gpu} in this experiments')
build/lib/mmdet/apis/train.py:        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
build/lib/mmdet/apis/train.py:            cfg.data.samples_per_gpu,
build/lib/mmdet/apis/train.py:            cfg.data.workers_per_gpu,
build/lib/mmdet/apis/train.py:            # cfg.gpus will be ignored if distributed
build/lib/mmdet/apis/train.py:            len(cfg.gpu_ids),
build/lib/mmdet/apis/train.py:    # put model on gpus
build/lib/mmdet/apis/train.py:            model.cuda(),
build/lib/mmdet/apis/train.py:            device_ids=[torch.cuda.current_device()],
build/lib/mmdet/apis/train.py:            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
build/lib/mmdet/apis/train.py:        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
build/lib/mmdet/apis/train.py:        if val_samples_per_gpu > 1:
build/lib/mmdet/apis/train.py:            samples_per_gpu=val_samples_per_gpu,
build/lib/mmdet/apis/train.py:            workers_per_gpu=cfg.data.workers_per_gpu,
build/lib/mmdet/apis/__init__.py:from .test import multi_gpu_test, single_gpu_test
build/lib/mmdet/apis/__init__.py:    'multi_gpu_test', 'single_gpu_test'

```
