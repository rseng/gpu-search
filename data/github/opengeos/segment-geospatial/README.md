# https://github.com/opengeos/segment-geospatial

```console
docs/usage.md:device = 'cuda' if torch.cuda.is_available() else 'cpu'
docs/installation.md:If your system has a GPU, but the above commands do not install the GPU version of pytorch, you can force the installation of the GPU version of pytorch using the following command:
docs/installation.md:mamba install -c conda-forge segment-geospatial "pytorch=*=cuda*"
docs/installation.md:To enable GPU for segment-geospatial, run the following command to run a short benchmark on your GPU:
docs/installation.md:docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
docs/installation.md:        -device=<d>       (where d=0,1,2.... for the CUDA device to use)
docs/installation.md:        -numdevices=<i>   (where i=(number of CUDA devices > 0) to use for simulation)
docs/installation.md:        -compare          (compares simulation results running once on the default GPU and once on the CPU)
docs/installation.md:NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
docs/installation.md:GPU Device 0: "Turing" with compute capability 7.5
docs/installation.md:> Compute 7.5 CUDA device: [Quadro RTX 5000]
docs/installation.md:nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown.
docs/installation.md:sudo docker run --rm -it --gpus=all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
docs/installation.md:docker run -it -p 8888:8888 --gpus=all giswqs/segment-geospatial:latest
docs/index.md:The Segment Anything Model is computationally intensive, and a powerful GPU is recommended to process large datasets. It is recommended to have a GPU with at least 8 GB of GPU memory. You can utilize the free GPU resources provided by Google Colab. Alternatively, you can apply for [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research), which offers cloud credits to support academic research. If you are in the Greater China region, apply for the AWS Cloud Credit [here](https://aws.amazon.com/cn/events/educate_cloud/research-credits).
README.md:If your system has a GPU, but the above commands do not install the GPU version of pytorch, you can force the installation of the GPU version of pytorch using the following command:
README.md:mamba install -c conda-forge segment-geospatial "pytorch=*=cuda*"
README.md:The Segment Anything Model is computationally intensive, and a powerful GPU is recommended to process large datasets. It is recommended to have a GPU with at least 8 GB of GPU memory. You can utilize the free GPU resources provided by Google Colab. Alternatively, you can apply for [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research), which offers cloud credits to support academic research. If you are in the Greater China region, apply for the AWS Cloud Credit [here](https://aws.amazon.com/cn/events/educate_cloud/research-credits).
samgeo/hq_sam.py:            device (str, optional): The device to use. It can be one of the following: cpu, cuda.
samgeo/hq_sam.py:                Defaults to None, which will use cuda if available.
samgeo/hq_sam.py:        # Use cuda if available
samgeo/hq_sam.py:            device = "cuda" if torch.cuda.is_available() else "cpu"
samgeo/hq_sam.py:            if device == "cuda":
samgeo/hq_sam.py:                torch.cuda.empty_cache()
samgeo/hq_sam.py:                    )  # If mask is on GPU, use .cpu() before .numpy()
samgeo/hq_sam.py:    def clear_cuda_cache(self):
samgeo/hq_sam.py:        """Clear the CUDA cache."""
samgeo/hq_sam.py:        if torch.cuda.is_available():
samgeo/hq_sam.py:            torch.cuda.empty_cache()
samgeo/samgeo2.py:            device (Optional[str]): The device to use (e.g., "cpu", "cuda", "mps"). Defaults to None.
samgeo/samgeo2.py:                by the model. Higher numbers may be faster but use more GPU memory.
samgeo/samgeo2.py:                    )  # If mask is on GPU, use .cpu() before .numpy()
samgeo/fast_sam.py:            device (str, optional): The device to use. Defaults to "cuda" if available, otherwise "cpu".
samgeo/fast_sam.py:        # Use cuda if available
samgeo/fast_sam.py:            device = "cuda" if torch.cuda.is_available() else "cpu"
samgeo/fast_sam.py:            if device == "cuda":
samgeo/fast_sam.py:                torch.cuda.empty_cache()
samgeo/samgeo.py:            device (str, optional): The device to use. It can be one of the following: cpu, cuda.
samgeo/samgeo.py:                Defaults to None, which will use cuda if available.
samgeo/samgeo.py:        # Use cuda if available
samgeo/samgeo.py:            device = "cuda" if torch.cuda.is_available() else "cpu"
samgeo/samgeo.py:            if device == "cuda":
samgeo/samgeo.py:                torch.cuda.empty_cache()
samgeo/samgeo.py:                    )  # If mask is on GPU, use .cpu() before .numpy()
samgeo/samgeo.py:    def clear_cuda_cache(self):
samgeo/samgeo.py:        """Clear the CUDA cache."""
samgeo/samgeo.py:        if torch.cuda.is_available():
samgeo/samgeo.py:            torch.cuda.empty_cache()
samgeo/text_sam.py:        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
samgeo/text_sam.py:                    )  # If mask is on GPU, use .cpu() before .numpy()
samgeo/common.py:    """Choose a device (CPU or GPU) for deep learning.
samgeo/common.py:        empty_cache (bool): Whether to empty the CUDA cache if a GPU is used. Defaults to True.
samgeo/common.py:    if torch.cuda.is_available():
samgeo/common.py:        device = torch.device("cuda")
samgeo/common.py:    if device.type == "cuda":
samgeo/common.py:            torch.cuda.empty_cache()
samgeo/common.py:        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
samgeo/common.py:        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
samgeo/common.py:        if torch.cuda.get_device_properties(0).major >= 8:
samgeo/common.py:            torch.backends.cuda.matmul.allow_tf32 = True
samgeo/common.py:                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "

```
