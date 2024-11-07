# https://github.com/MedMaxLab/selfEEG

```console
setup.py:        "Environment :: GPU",
docs/faq.rst:1) **Does selfEEG support training on GPUs for MacOS devices?**
docs/faq.rst:The library is built on top of PyTorch, which support training on GPUs
docs/faq.rst:still not cover all the functionalities implemented in CUDA (a coverage matrix can be
docs/faq.rst:2) **Is an augmentation always faster on GPU devices?**
docs/faq.rst:Most of the time, augmentations executed on GPUs are faster compared to one on CPUs.
docs/faq.rst:time of augmentations: the GPU device (cuda or mps), the ``batch_equal`` argument,
docs/faq.rst:run on the Padova Neuroscience Center Server (GPU Tesla V100) with a 3D array
docs/_static/bench_table.csv:,Numpy,Numpy,Torch,Torch,Torch GPU,Torch GPU
docs/selfeeg.utils.rst:This module simply gathers functions and classes for various purposes. For example, you can find a torch implementation of the Scipy's pchip interpolation function (used for resampling) or pytorch EEG scaler with a soft clipping option. Both the cited functions are compatible with GPU tensors.
docs/index.rst:   with full GPU support, as well asother classes designed to combine them in more
docs/index.rst:packages. If you have CUDA installed on your system, we strongly encourage you to
docs/index.rst:which varies depending on your OS and CUDA versions; then install selfEEG.
docs/selfeeg.augmentation.rst:  and torch tensors moved to both CPU or GPU devices.
docs/CONTRIBUTING.rst:selfEEG functionalities also on a GPU device.
selfeeg/dataloading/load.py:        specific requirements since float32 are faster on GPU devices.
selfeeg/losses/losses.py:        N = logits.shape[0]  # batch size per GPU
selfeeg/ssl/base.py:        #    a duplicate on the GPU
selfeeg/ssl/base.py:        # Original command: model is not moved but a copy is created on the gpu
selfeeg/ssl/base.py:            # If device is None simply use gpu
selfeeg/utils/utils.py:        last dimension. Tensors can also be placed in a GPU.
selfeeg/utils/utils.py:        dimension. Tensors can also be placed in a GPU.
selfeeg/utils/utils.py:    This function is compatible with GPU devices.
Paper/paper.md:Most of the functionality offered by selfEEG can be executed both on GPUs and CPUs, expanding its usability beyond the self-supervised learning area.
test/README.md:2. Padova Neuroscience Center Server, Ubuntu 18.04.1, Tesla V100 GPU
test/README.md:3. Custom built PC, Windows 10 22H2, NVIDIA RTX 2080 super
test/EEGself/losses/losses_test.py:        elif torch.cuda.is_available():
test/EEGself/losses/losses_test.py:            cls.device = torch.device("cuda")
test/EEGself/losses/losses_test.py:            print("Found other device: testing module with both cpu and gpu")
test/EEGself/losses/losses_test.py:            print("Didn't found cuda device: testing module with only cpu")
test/EEGself/models/zoo_test.py:        elif torch.cuda.is_available():
test/EEGself/models/zoo_test.py:            cls.device = torch.device("cuda")
test/EEGself/models/zoo_test.py:            print("Found gpu device: testing module with both cpu and gpu")
test/EEGself/models/zoo_test.py:            print("Didn't found cuda device: testing module with only cpu")
test/EEGself/models/layers_test.py:        elif torch.cuda.is_available():
test/EEGself/models/layers_test.py:            cls.device = torch.device("cuda")
test/EEGself/models/layers_test.py:            print("Found gpu device: testing module with both cpu and gpu")
test/EEGself/models/layers_test.py:            print("Didn't found cuda device: testing module with only cpu")
test/EEGself/ssl/ssl_test.py:        elif torch.cuda.is_available():
test/EEGself/ssl/ssl_test.py:            cls.device = torch.device("cuda")
test/EEGself/ssl/ssl_test.py:            print("Found cuda device: testing module on it")
test/EEGself/ssl/ssl_test.py:            print("Didn't found cuda device: testing module on cpu")
test/EEGself/augmentation/functional_test.py:        elif torch.cuda.is_available():
test/EEGself/augmentation/functional_test.py:            cls.device = torch.device("cuda")
test/EEGself/augmentation/functional_test.py:            print("Found gpu device: testing module on it")
test/EEGself/augmentation/functional_test.py:            print("Didn't found cuda device: testing module on cpu")
test/EEGself/augmentation/functional_test.py:            cls.x1gpu = torch.clone(cls.x1).to(device=device)
test/EEGself/augmentation/functional_test.py:            cls.x2gpu = torch.clone(cls.x2).to(device=device)
test/EEGself/augmentation/functional_test.py:            cls.x3gpu = torch.clone(cls.x3).to(device=device)
test/EEGself/augmentation/functional_test.py:            cls.x4gpu = torch.clone(cls.x4).to(device=device)
test/EEGself/augmentation/functional_test.py:            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
test/EEGself/augmentation/functional_test.py:            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu], "value": [1, 2.0, 4]}
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
test/EEGself/augmentation/functional_test.py:            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu]}
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:            aug_args = {"x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu], "order": [3, 5, 9]}
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/augmentation/functional_test.py:                "x": [self.x1gpu, self.x2gpu, self.x3gpu, self.x4gpu],
test/EEGself/utils/utils_test.py:        elif torch.cuda.is_available():
test/EEGself/utils/utils_test.py:            cls.device = torch.device("cuda")
test/EEGself/utils/utils_test.py:            print("Found gpu device: testing module with both cpu and gpu")
test/EEGself/utils/utils_test.py:            print("Didn't found cuda device: testing module with only cpu")
test/EEGself/utils/utils_test.py:            xgpu = torch.clone(x).to(device=self.device)
test/EEGself/utils/utils_test.py:            inplist = [x, xnp, xgpu]
test/EEGself/utils/utils_test.py:            xgpu = torch.clone(x).to(device=self.device)
test/EEGself/utils/utils_test.py:            inplist = [x, xnp, xgpu]
README.md:3. **augmentation** - collection of data augmentation with fully support on GPU
README.md:If you have CUDA installed on your system, we strongly encourage you to first
README.md:right configuration, which varies depending on your OS and CUDA versions;
extra_material/bench_table.csv:,Numpy Array no BE,Numpy Array BE,Torch Tensor no BE,Torch Tensor BE,Torch Tensor GPU no BE,Torch Tensor GPU BE
extra_material/README.md:1. **Augmentation_benchmark.py**: A file that run a benchmark test for the augmentation module. The file run 10 calls per function with multiple configurations (tensor/numpy array, GPU/CPU, batch_equal True/False), leaving NaN for those configurations not executable. The total execution time per configuration will be stored in the bench_table.csv file. The file also accepts an optional arguments to set the number of repetition of the 10 calls per configuration (total calls will be 10*repetition). Higher the number repetitions, higher the total execution time. To run the code, simply use the following command, changing X with an integer for the number of repetitions (use -h instead of -r X for additional help)
extra_material/Augmentation_benchmark.py:values if not possible (e.g., functions with no batch_equal arg or not available GPU device):
extra_material/Augmentation_benchmark.py:5. Torch Tensor on GPU with no batch equal
extra_material/Augmentation_benchmark.py:6. Torch Tensor on GPU with batch equal
extra_material/Augmentation_benchmark.py:    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
extra_material/Augmentation_benchmark.py:sup_torch_gpu = """
extra_material/Augmentation_benchmark.py:    device= torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
extra_material/Augmentation_benchmark.py:    bench_dict["add_band_noise"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["add_eeg_artifact"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["add_eeg_artifact"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["add_gaussian_noise"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["add_noise_SNR"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["change_ref"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["channel_dropout"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["channel_dropout"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["crop_and_resize"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["crop_and_resize"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["filter_bandpass"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["filter_bandstop"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["filter_highpass"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["filter_lowpass"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["flip_horizontal"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["flip_vertical"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["masking"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["masking"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["moving_avg"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permutation_signal"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permutation_signal"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permute_channels_network"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permute_channels_network"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permute_channels"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["permute_channels"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:if device.type == "cuda":
extra_material/Augmentation_benchmark.py:    bench_dict["random_FT_phase"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["random_FT_phase"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["random_slope_scale"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["random_slope_scale"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["scaling"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["scaling"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:if device.type == "cuda":
extra_material/Augmentation_benchmark.py:    bench_dict["shift_frequency"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["shift_frequency"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["shift_horizontal"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["shift_horizontal"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["shift_vertical"][5] = timeit.timeit(s, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["warp_signal"][4] = timeit.timeit(s_false, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:    bench_dict["warp_signal"][5] = timeit.timeit(s_true, sup_torch_gpu, number=n)
extra_material/Augmentation_benchmark.py:        "Torch Tensor GPU no BE",
extra_material/Augmentation_benchmark.py:        "Torch Tensor GPU BE",
CONTRIBUTING.md:a GPU device.

```
