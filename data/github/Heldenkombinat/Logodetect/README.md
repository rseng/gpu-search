# https://github.com/Heldenkombinat/Logodetect

```console
tests/test_utils.py:from logodetect.utils import open_and_resize, image_to_gpu_tensor, clean_name, save_df
tests/test_utils.py:def test_image_to_gpu():
tests/test_utils.py:    image_to_gpu_tensor(
tests/test_utils.py:    )  # can't use cuda etc. here, as not every system will have it
README.md:[2] He et. al. [Deep Residual Learning for Image Recognition](http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) (2016)\
training/configs/recognizer_cifar10.json:        "device": "cuda:1",
training/configs/recognizer_stacked.json:        "device": "cuda:1",
training/configs/detector.json:        "device": "cuda:2",
training/configs/recognizer_hyperparam-search.json:        "device": "cuda:1",
training/configs/recognizer_one-class.json:        "device": "cuda:1",
training/loreta/trainers.py:            # Move data to GPU:
training/loreta/trainers.py:            images, targets = self.move_to_gpu(images, targets)
training/loreta/trainers.py:            # Move data to GPU:
training/loreta/trainers.py:            images, targets = self.move_to_gpu(images, targets)
training/loreta/trainers.py:            torch.cuda.synchronize()
training/loreta/trainers.py:    def move_to_gpu(self, images, targets):
training/loreta/utils.py:def image_to_gpu_tensor(image, device=None):
training/loreta/utils.py:    "Checks if image is valid and moves it to the GPU."
training/loreta/utils.py:    DEVICE = CONFIG["general"]["device"]  # {'cuda:1', 'cpu'}
training/loreta/pytorch/detect_utils.py:DEVICE = CONFIG["general"]["device"]  # {'cuda:1', 'cpu'}
training/loreta/pytorch/detect_utils.py:        if torch.cuda.is_available():
training/loreta/pytorch/detect_utils.py:                if torch.cuda.is_available():
training/loreta/pytorch/detect_utils.py:                            memory=torch.cuda.max_memory_allocated() / MB,
training/loreta/pytorch/detect_utils.py:        args.gpu = int(os.environ["LOCAL_RANK"])
training/loreta/pytorch/detect_utils.py:        args.gpu = args.rank % torch.cuda.device_count()
training/loreta/pytorch/detect_utils.py:    torch.cuda.set_device(args.gpu)
training/loreta/pytorch/detect_utils.py:    args.dist_backend = "nccl"
logodetect/detectors/faster_rcnn.py:from logodetect.utils import image_to_gpu_tensor
logodetect/detectors/faster_rcnn.py:        image = image_to_gpu_tensor(image, self.config.get("DETECTOR_DEVICE"))
logodetect/classifiers/siamese.py:from logodetect.utils import clean_name, open_and_resize, image_to_gpu_tensor
logodetect/classifiers/siamese.py:            image_gpu = image_to_gpu_tensor(image, self.config.get("CLASSIFIER_DEVICE"))
logodetect/classifiers/siamese.py:            self.exemplars_imgs.append(image_gpu)
logodetect/classifiers/siamese.py:                aug_image_gpu = image_to_gpu_tensor(
logodetect/classifiers/siamese.py:                self.exemplars_imgs.append((aug_image_gpu))
logodetect/classifiers/siamese.py:            detection = image_to_gpu_tensor(crop, self.config.get("CLASSIFIER_DEVICE"))
logodetect/constants.py:- and the devices you want to use (CPU, GPU, etc.)
logodetect/constants.py:        "DEVICE": "cpu",  # {cpu, cuda:1, cuda:2, ...}
logodetect/utils.py:    """Checks if image is valid and moves it to the GPU.
logodetect/utils.py:def image_to_gpu_tensor(image: Image.Image, device: str) -> torch.Tensor:
logodetect/utils.py:    """Checks if image is valid and moves it to the GPU.

```
