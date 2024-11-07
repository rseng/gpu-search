# https://github.com/masadcv/FastGeodis

```console
setup.py:FORCE_CUDA = os.getenv("FORCE_CUDA", "0") == "1"
setup.py:BUILD_CPP = BUILD_CUDA = False
setup.py:    from torch.utils.cpp_extension import CUDA_HOME, CUDAExtension
setup.py:    BUILD_CUDA = (CUDA_HOME is not None) if torch.cuda.is_available() else FORCE_CUDA
setup.py:        f"BUILD_CPP={BUILD_CPP}, BUILD_CUDA={BUILD_CUDA}, TORCH_VERSION={TORCH_VERSION}."
setup.py:    source_cuda = glob.glob(os.path.join(ext_dir, "**", "*.cu"), recursive=True)
setup.py:    if BUILD_CUDA:
setup.py:        extension = CUDAExtension
setup.py:        sources += source_cuda
setup.py:        define_macros += [("WITH_CUDA", None)]
setup.py:    description="Fast Implementation of Generalised Geodesic Distance Transform for CPU (OpenMP) and GPU (CUDA)",
docs/source/getting_started.rst:**FastGeodis** provides efficient CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both.
docs/source/getting_started.rst:or (on conda environment with existing installation of PyTorch with CUDA)
docs/source/getting_started.rst:In addition, for compilation and execution on GPU, the **FastGeodis** package requires a CUDA installation compatible with installed PyTorch version. 
docs/source/getting_started.rst:    device = "cuda" if torch.cuda.is_available() else "cpu"
docs/source/getting_started.rst:    device = "cuda" if torch.cuda.is_available() else "cpu"
docs/source/methodology.rst:Despite existing open-source implementation of distance transforms :cite:p:`tensorflow2015-whitepaper,eucildeantdimpl,geodistk`, open-source implementations of efficient Geodesic distance transform algorithms :cite:p:`criminisiinteractive,weber2008parallel` on CPU and GPU do not exist. However, efficient CPU :cite:p:`eucildeantdimpl` and GPU :cite:p:`tensorflow2015-whitepaper` implementations exist for Euclidean distance transform. To the best of our knowledge, **FastGeodis** is the first open-source implementation of efficient Geodesic distance transform :cite:p:`criminisiinteractive`, achieving up to 20x speed-up on CPU and up to 74x speed-up on GPU as compared to existing open-source libraries :cite:p:`geodistk`. It also provides efficient implementation of Euclidean distance transform. In addition, it is the first open-source implementation of generalised Geodesic distance transform and Geodesic Symmetric Filtering (GSF) proposed in :cite:p:`criminisi2008geos`. 
docs/source/methodology.rst:The ability to efficiently compute Geodesic and Euclidean distance transforms can significantly enhance distance transform applications especially for training deep learning models that utilise distance transforms :cite:p:`wang2018deepigeos`. It will improve prototyping, experimentation, and deployment of such methods, where efficient computation of distance transforms has been a limiting factor. In 3D medical imaging problems, efficient computation of distance transforms will lead to significant speed-ups, enabling online learning applications for better processing/labelling/inference from volumetric datasets :cite:p:`asad2022econet`.  In addition, **FastGeodis** provides efficient implementation for both CPUs and GPUs hardware and hence will enable efficient use of a wide range of hardware devices. 
docs/source/methodology.rst:**FastGeodis** package is implemented using **PyTorch** :cite:p:`NEURIPS2019_9015` utilising OpenMP for CPU and CUDA for GPU parallelisation of the algorithm. It is accessible as a python package, that can be installed across different operating systems and devices. A comprehensive documentation and a range of examples are provided for understanding the usage of the package on 2D and 3D data using CPU or GPU. The provided examples include 2D/3D examples for Geodesic, Euclidean, Signed Geodesic distance transform as well as computing Geodesic symmetric filtering (GSF) that is essential first step in implementing interactive segmentation method from :cite:p:`criminisi2008geos`. 
docs/source/methodology.rst:We implement both 2D and 3D parallelisable generalised Geodesic distance transform algorithms from :cite:p:`criminisiinteractive` on both CPU (OpenMP) and GPU (CUDA). The 2D algorithm works by computing distance propagation in one row at a time. This is a hard constraint because compute for each row is dependent on computed distances in previous rows from previous invocation. Consider an example of an image with 4 x 6 dimension. Then a successful full iteration of our method would involve going through the rows one by one as follows:
docs/source/methodology.rst:As can be seen, this involves top-down and left-right passes. How we implement left-right is by reusing top-down code and transposing the data instead. Please note that for each step, only the pixels highlighted in green color can be computed (as they have the data available from previous row). It is this row that we split into multiple threads using an underlying hardware (CPU or GPU). This parallelisation enables speed-up as compared to non-parallelisable CPU implementations, e.g. in :cite:p:`geodistk` which implements raster scan algorithm from :cite:p:`toivanen1996new`. 
docs/source/methodology.rst:Going beyond 2D images, we also implement the parallelisable Geodesic distance transform for 3D data. We provide both CPU (OpenMP) and GPU (CUDA) optimised implementations. Our 3D implementation operates on the same principle (we can process one plane at a time). However, in 3D case, since we have more data, we can utilise more compute on GPU and process a plane in parallel, however we still have the data dependency constraints that prevent us from processing all planes together.
docs/source/methodology.rst::cite:p:`weber2008parallel` presents a further optimised approach for computing Geodesic distance transforms on GPUs, however this method is protected by multiple patents and hence is not suitable for open-source implementation in **FastGeodis** package.
docs/source/methodology.rst:FastGeodis (CPU/GPU) is compared with existing GeodisTK (https://github.com/taigw/GeodisTK) in terms of execution speed as well as accuracy. All our experiments were evaluated on Nvidia GeForce Titan X (12 GB) with 6-Core Intel Xeon E5-1650 CPU. We present our results below:
docs/source/methodology.rst:It can be observed that for 2D images, **FastGeodis** leads to a speed-up of upto 20x on CPU and upto 55x on GPU.
docs/source/methodology.rst:For 3D images, **FastGeodis** leads to a speed-up of upto 3x on CPU and upto 74x on GPU.
docs/source/index.rst:**FastGeodis** provides efficient CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both.
docs/source/index.rst:The above raster scan method can be parallelised for each row/plane on an available device (CPU or GPU). This leads to significant speed up as compared to existing non-parallelised raster scan implementations (e.g. https://github.com/taigw/GeodisTK). Python interface is provided (using PyTorch) for enabling its use in deep learning and image processing pipelines.
docs/source/experiment2d.csv:Device,CPU,CPU,GPU,CPU,GPU
docs/source/usage_examples.rst:    device = "cuda" if torch.cuda.is_available() else "cpu"
docs/source/usage_examples.rst:Note: the above example execute using CPU with :code:`device = "cpu"`. To change execution device to GPU use :code:`device="cuda"`.
docs/source/experiment3d.csv:Device,CPU,CPU,GPU,CPU,GPU
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @skip_if_no_cuda
tests/test_fastgeodis.py:        device = "cuda"
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastgeodis.py:    @skip_if_no_cuda
tests/test_fastgeodis.py:        device = "cuda"
tests/test_fastgeodis.py:    @run_cuda_if_available
tests/test_fastmarch.py:    @run_cuda_if_available
tests/test_fastmarch.py:    @run_cuda_if_available
tests/test_fastmarch.py:    @run_cuda_if_available
tests/test_fastmarch.py:    @run_cuda_if_available
tests/test_fastmarch.py:    @run_cuda_if_available
tests/utils.py:DEVICES_TO_RUN = ["cpu", "cuda"]
tests/utils.py:CONF_2D_CUDA = [("cuda", 2, bas) for bas in [16, 32, 64]]
tests/utils.py:CONF_2D = CONF_2D_CPU + CONF_2D_CUDA
tests/utils.py:CONF_3D_CUDA = [("cuda", 3, bas) for bas in [16, 32, 64]]
tests/utils.py:CONF_3D = CONF_3D_CPU + CONF_3D_CUDA
tests/utils.py:def skip_if_no_cuda(obj):
tests/utils.py:    return unittest.skipUnless(torch.cuda.is_available(), "Skipping CUDA-based tests")(
tests/utils.py:def run_cuda_if_available(fn):
tests/utils.py:        if args[1] == "cuda":
tests/utils.py:            if torch.cuda.is_available():
tests/utils.py:                raise unittest.SkipTest("skipping as cuda device not found")
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_toivanen.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
tests/test_pixelqueue.py:    @run_cuda_if_available
samples/demo2d_signed.py:    device = "cuda" if torch.cuda.is_available() else None
samples/demo2d_signed.py:        fastraster_output_gpu = np.squeeze(
samples/demo2d_signed.py:        fastraster_time_gpu = time.time() - tic
samples/demo2d_signed.py:        print("FastGeodis GPU raster: {:.6f} s".format(fastraster_time_gpu))
samples/demo2d_signed.py:        plt.imshow(fastraster_output_gpu)
samples/demo2d_signed.py:        plt.title("(f) FastGeodis (gpu) | ({:.4f} s)".format(fastraster_time_gpu))
samples/demo2d_signed.py:            abs(fastmarch_output - fastraster_output_gpu)
samples/demo2d_signed.py:            "(h) Fast Marching vs. FastGeodis (gpu)\ndiff: max: {:.4f} | min: {:.4f}".format(
samples/demo2d_signed.py:            plt.title("Joint histogram\nFast Marching (cpu) vs. FastGeodis (gpu)")
samples/demo2d_signed.py:                fastraster_output_gpu.flatten(),
samples/demo2d_signed.py:            plt.ylabel("FastGeodis (gpu)")
samples/simpledemo2d.py:device = "cuda" if torch.cuda.is_available() else "cpu"
samples/demo3d_signed.py:        "cuda" if input_image_pt.shape[1] == 1 and torch.cuda.is_available() else None
samples/demo3d_signed.py:        fastraster_output_gpu = np.squeeze(
samples/demo3d_signed.py:        fastraster_time_gpu = time.time() - tic
samples/demo3d_signed.py:        print("FastGeodis GPU raster: {:.6f} s".format(fastraster_time_gpu))
samples/demo3d_signed.py:        fastraster_output_gpu_slice = fastraster_output_gpu[10]
samples/demo3d_signed.py:        plt.imshow(fastraster_output_gpu_slice)
samples/demo3d_signed.py:        plt.title("(f) FastGeodis (gpu) | ({:.4f} s)".format(fastraster_time_gpu))
samples/demo3d_signed.py:            abs(fastmarch_output - fastraster_output_gpu)
samples/demo3d_signed.py:        diff_vol = fastmarch_output - fastraster_output_gpu
samples/demo3d_signed.py:            "(h) Fast Marching vs. FastGeodis (gpu)\ndiff: max: {:.4f} | min: {:.4f}".format(
samples/demo3d_signed.py:                fastraster_output_gpu.flatten(),
samples/demo3d_signed.py:            plt.ylabel("FastGeodis (gpu)")
samples/simpledemo3d.py:device = "cuda" if torch.cuda.is_available() else "cpu"
samples/simpledemo2d_signed.py:device = "cuda" if torch.cuda.is_available() else "cpu"
samples/demo3d.py:        "cuda" if input_image_pt.shape[1] == 1 and torch.cuda.is_available() else None
samples/demo3d.py:        fastraster_output_gpu = np.squeeze(
samples/demo3d.py:        fastraster_time_gpu = time.time() - tic
samples/demo3d.py:        print("FastGeodis GPU raster: {:.6f} s".format(fastraster_time_gpu))
samples/demo3d.py:        fastraster_output_gpu_slice = fastraster_output_gpu[10]
samples/demo3d.py:        plt.imshow(fastraster_output_gpu_slice)
samples/demo3d.py:        plt.title("(f) FastGeodis (gpu) | ({:.4f} s)".format(fastraster_time_gpu))
samples/demo3d.py:            abs(fastmarch_output - fastraster_output_gpu)
samples/demo3d.py:        diff_vol = fastmarch_output - fastraster_output_gpu
samples/demo3d.py:            "(h) Fast Marching vs. FastGeodis (gpu)\ndiff: max: {:.4f} | min: {:.4f}".format(
samples/demo3d.py:                fastraster_output_gpu.flatten(),
samples/demo3d.py:            plt.ylabel("FastGeodis (gpu)")
samples/test_speed_benchmark_geodistk.py:def generalised_geodesic2d_raster_gpu(I, S, v, lamb, iter):
samples/test_speed_benchmark_geodistk.py:def generalised_geodesic3d_raster_gpu(I, S, spacing, v, lamb, iter):
samples/test_speed_benchmark_geodistk.py:func_to_test_2d = [generalised_geodesic_distance_2d, generalised_geodesic2d_raster_cpu, generalised_geodesic2d_raster_gpu]
samples/test_speed_benchmark_geodistk.py:func_to_test_3d = [generalised_geodesic_distance_3d, generalised_geodesic3d_raster_cpu, generalised_geodesic3d_raster_gpu]
samples/test_speed_benchmark_geodistk.py:            if 'gpu' in func.__name__:
samples/test_speed_benchmark_geodistk.py:                image = image.to('cuda').contiguous()
samples/test_speed_benchmark_geodistk.py:                seed = seed.to('cuda').contiguous()
samples/test_speed_benchmark_geodistk.py:                elif 'gpu' in func.__name__ and torch.cuda.is_available():
samples/test_speed_benchmark_geodistk.py:            if 'gpu' in func.__name__:
samples/test_speed_benchmark_geodistk.py:                image = image.to('cuda').contiguous()
samples/test_speed_benchmark_geodistk.py:                seed = seed.to('cuda').contiguous()
samples/test_speed_benchmark_geodistk.py:                elif 'gpu' in func.__name__ and torch.cuda.is_available():
samples/test_speed_benchmark_geodistk.py:        elif 'gpu' in key:
samples/test_speed_benchmark_geodistk.py:            plt.plot(sizes, time_taken_dict[key], 'g-o', label='FastGeodis (gpu)')
samples/test_speed_benchmark_toivanen.py:def generalised_geodesic2d_raster_gpu(I, S, v, lamb, iter):
samples/test_speed_benchmark_toivanen.py:def generalised_geodesic3d_raster_gpu(I, S, spacing, v, lamb, iter):
samples/test_speed_benchmark_toivanen.py:    generalised_geodesic2d_raster_gpu,
samples/test_speed_benchmark_toivanen.py:    generalised_geodesic3d_raster_gpu,
samples/test_speed_benchmark_toivanen.py:            if "gpu" in func.__name__:
samples/test_speed_benchmark_toivanen.py:                image = image.to("cuda").contiguous()
samples/test_speed_benchmark_toivanen.py:                seed = seed.to("cuda").contiguous()
samples/test_speed_benchmark_toivanen.py:                elif "gpu" in func.__name__ and torch.cuda.is_available():
samples/test_speed_benchmark_toivanen.py:            if "gpu" in func.__name__:
samples/test_speed_benchmark_toivanen.py:                image = image.to("cuda").contiguous()
samples/test_speed_benchmark_toivanen.py:                seed = seed.to("cuda").contiguous()
samples/test_speed_benchmark_toivanen.py:                elif "gpu" in func.__name__ and torch.cuda.is_available():
samples/test_speed_benchmark_toivanen.py:        elif "gpu" in key and "raster" in key:
samples/test_speed_benchmark_toivanen.py:            plt.plot(sizes, time_taken_dict[key], "g-o", label="FastGeodis (gpu)")
samples/simpledemo3d_profile.py:device = "cuda" if torch.cuda.is_available() else "cpu"
samples/demo2d.py:    device = "cuda" if torch.cuda.is_available() else None
samples/demo2d.py:        fastraster_output_gpu = np.squeeze(
samples/demo2d.py:        fastraster_time_gpu = time.time() - tic
samples/demo2d.py:        print("FastGeodis GPU raster: {:.6f} s".format(fastraster_time_gpu))
samples/demo2d.py:        plt.imshow(fastraster_output_gpu)
samples/demo2d.py:        plt.title("(f) FastGeodis (gpu) | ({:.4f} s)".format(fastraster_time_gpu))
samples/demo2d.py:            abs(fastmarch_output - fastraster_output_gpu)
samples/demo2d.py:            "(h) Fast Marching vs. FastGeodis (gpu)\ndiff: max: {:.4f} | min: {:.4f}".format(
samples/demo2d.py:            plt.title("Joint histogram\nFast Marching (cpu) vs. FastGeodis (gpu)")
samples/demo2d.py:                fastraster_output_gpu.flatten(),
samples/demo2d.py:            plt.ylabel("FastGeodis (gpu)")
samples/simpledemo2d_profile.py:device = "cuda" if torch.cuda.is_available() else "cpu"
samples/simpledemo3d_signed.py:device = "cuda" if torch.cuda.is_available() else "cpu"
figures/experiment_2d.json:    "generalised_geodesic2d_raster_gpu": [
figures/experiment_3d.json:    "generalised_geodesic3d_raster_gpu": [
README.md:This repository provides CPU (OpenMP) and GPU (CUDA) implementations of Generalised Geodesic Distance Transform in PyTorch for 2D and 3D input data based on parallelisable raster scan ideas from [1]. It includes methods for computing Geodesic, Euclidean distance transform and mixture of both. 
README.md:The above raster scan method can be parallelised for each row/plane on an available device (CPU or GPU). This leads to significant speed up as compared to existing non-parallelised raster scan implementations (e.g. [https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)). Python interface is provided (using PyTorch) for enabling its use in deep learning and image processing pipelines.
README.md:> The raster scan based implementation provides a balance towards speed rather than accuracy of Geodesic distance transform and hence results in efficient hardware utilisation. On the other hand, in case of Euclidean distance transform, exact results can be achieved with other packages (albeit not on necessarilly on GPU) [6, 7, 8]
README.md:or (on conda environments with existing installation of PyTorch with CUDA)
README.md:## Optimised Fast Implementations for GPU/CPU based on [1]
README.md:| Fast Generalised Geodesic Distance 2D   |  Paralellised generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.generalised_geodesic2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic2d)         |
README.md:| Fast Generalised Geodesic Distance 3D   |  Paralellised generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.generalised_geodesic3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.generalised_geodesic3d)         |
README.md:| Fast Signed Generalised Geodesic Distance 2D   |  Paralellised signed generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.signed_generalised_geodesic2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic2d)         |
README.md:| Fast Signed Generalised Geodesic Distance 3D   |  Paralellised signed generalised geodesic distance transform for CPU/GPU [1]          |      [FastGeodis.signed_generalised_geodesic3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.signed_generalised_geodesic3d)         |
README.md:| Fast Geodesic Symmetric Filtering 2D   |  Paralellised geodesic symmetric filtering for CPU/GPU [2]          |      [FastGeodis.GSF2d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF2d)         |
README.md:| Fast Geodesic Symmetric Filtering 3D   |  Paralellised geodesic symmetric filtering for CPU/GPU [2]          |      [FastGeodis.GSF3d](https://fastgeodis.readthedocs.io/en/latest/api_docs.html#FastGeodis.GSF3d)         |
README.md:device = "cuda" if torch.cuda.is_available() else "cpu"
README.md:device = "cuda" if torch.cuda.is_available() else "cpu"
README.md:FastGeodis (CPU/GPU) is compared with existing GeodisTK ([https://github.com/taigw/GeodisTK](https://github.com/taigw/GeodisTK)) in terms of execution speed as well as accuracy.
paper/paper.md:Geodesic and Euclidean distance transforms have been widely used in a number of applications where distance from a set of reference points is computed. Methods from recent years have shown effectiveness in applying the Geodesic distance transform to interactively annotate 3D medical imaging data [@wang2018deepigeos; @criminisi2008geos]. The Geodesic distance transform enables providing segmentation labels, i.e., voxel-wise labels, for different objects of interests. Despite existing methods for efficient computation of the Geodesic distance transform on GPU and CPU devices [@criminisiinteractive; @criminisi2008geos; @weber2008parallel; @toivanen1996new], an open-source implementation of such methods on the GPU does not exist. 
paper/paper.md:The `FastGeodis` package provides an efficient implementation for computing Geodesic and Euclidean distance transforms (or a mixture of both), targeting efficient utilisation of CPU and GPU hardware. In particular, it implements the paralellisable raster scan method from @criminisiinteractive, where elements in a row (2D) or plane (3D) can be computed with parallel threads. This package is able to handle 2D as well as 3D data, where it achieves up to a 20x speedup on a CPU and up to a 74x speedup on a GPU as compared to an existing open-source library [@geodistk] that uses a non-parallelisable single-thread CPU implementation. The performance speedups reported here were evaluated using 3D volume data on an Nvidia GeForce Titan X (12 GB) with a 6-Core Intel Xeon E5-1650 CPU. Further in-depth comparison of performance improvements is discussed in the `FastGeodis` \href{https://fastgeodis.readthedocs.io/}{documentation}. 
paper/paper.md:Despite existing open-source implementation of distance transforms [@tensorflow2015-whitepaper; @eucildeantdimpl; @geodistk], open-source implementations of efficient Geodesic distance transform algorithms [@criminisiinteractive; @weber2008parallel] on CPUs and GPUs do not exist. However, efficient CPU [@eucildeantdimpl] and GPU [@tensorflow2015-whitepaper] implementations exist for Euclidean distance transform. To the best of our knowledge, `FastGeodis` is the first open-source implementation of efficient the Geodesic distance transform [@criminisiinteractive], achieving up to a 20x speedup on a CPU and up to a 74x speedup on a GPU as compared to existing open-source libraries [@geodistk]. It also provides an efficient implementation of the Euclidean distance transform. In addition, it is the first open-source implementation of generalised Geodesic distance transform and Geodesic Symmetric Filtering (GSF) as proposed in @criminisi2008geos. Apart from a method from @criminisiinteractive, @weber2008parallel present a further optimised approach for computing Geodesic distance transforms on GPUs. However, this method is protected by multiple patents [@bronstein2013parallel; @bronstein2015parallel; @bronstein2016parallel] and hence is not suitable for open-source implementation in the **FastGeodis** package.
paper/paper.md:The ability to efficiently compute Geodesic and Euclidean distance transforms can significantly enhance distance transform applications, especially for training deep learning models that utilise distance transforms [@wang2018deepigeos]. It will improve prototyping, experimentation, and deployment of such methods, where efficient computation of distance transforms has been a limiting factor. In 3D medical imaging problems, efficient computation of distance transforms will lead to significant speed-ups, enabling online learning applications for better processing/labelling/inference from volumetric datasets [@asad2022econet].  In addition, `FastGeodis` provides an efficient implementation for both CPUs and GPUs and hence will enable efficient use of a wide range of hardware devices. 
paper/paper.md:The `FastGeodis` package is implemented using `PyTorch` [@NEURIPS2019_9015], utilising OpenMP for CPU- and CUDA for GPU-parallelisation of the algorithm. It is accessible as a Python package that can be installed across different operating systems and devices. Comprehensive documentation and a range of examples are provided for understanding the usage of the package on 2D and 3D data using CPUs or GPUs. Two- and three-dimensional examples are provided for Geodesic, Euclidean, and Signed Geodesic distance transforms as well as for computing Geodesic Symmetric Filtering (GSF), the essential first step in implementing the interactive segmentation method described in @criminisi2008geos. A further in-depth overview of the implemented algorithm, along with evaluation on common 2D/3D data input sizes, is provided in the `FastGeodis` \href{https://fastgeodis.readthedocs.io/}{documentation}.
FastGeodis/fastgeodis.h:#ifdef WITH_CUDA
FastGeodis/fastgeodis.h:torch::Tensor generalised_geodesic2d_cuda(
FastGeodis/fastgeodis.h:torch::Tensor generalised_geodesic3d_cuda(
FastGeodis/common.h:    if (in.is_cuda())
FastGeodis/common.h:void check_cuda(const torch::Tensor &in)
FastGeodis/common.h:    if (!in.is_cuda())
FastGeodis/common.h:        throw std::invalid_argument("input is not on CUDA device, try using data.to('cuda') on input");
FastGeodis/fastgeodis_cuda.cu:#include <c10/cuda/CUDAGuard.h>
FastGeodis/fastgeodis_cuda.cu:#include <cuda.h>
FastGeodis/fastgeodis_cuda.cu:#include <cuda_runtime.h>
FastGeodis/fastgeodis_cuda.cu:// whether to use float* or Pytorch accessors in CUDA kernels
FastGeodis/fastgeodis_cuda.cu:__device__ float l1distance_cuda(const float &in1, const float &in2)
FastGeodis/fastgeodis_cuda.cu:                        l_dist = l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                            l_dist += l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                        l_dist = l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                            l_dist += l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:void geodesic_updown_pass_cuda(
FastGeodis/fastgeodis_cuda.cu:    // copy local distances to GPU __constant__ memory
FastGeodis/fastgeodis_cuda.cu:    cudaMemcpyToSymbol(local_dist2d, local_dist, sizeof(float) * 3);
FastGeodis/fastgeodis_cuda.cu:    // process each row in parallel with CUDA kernel
FastGeodis/fastgeodis_cuda.cu:torch::Tensor generalised_geodesic2d_cuda(
FastGeodis/fastgeodis_cuda.cu:    // std::cout << "Running with CUDA Device: " << device << std::endl;
FastGeodis/fastgeodis_cuda.cu:    c10::cuda::CUDAGuard device_guard(device);
FastGeodis/fastgeodis_cuda.cu:        geodesic_updown_pass_cuda(image_local, distance, l_grad, l_eucl);
FastGeodis/fastgeodis_cuda.cu:        geodesic_updown_pass_cuda(image_local, distance, l_grad, l_eucl);
FastGeodis/fastgeodis_cuda.cu:                            l_dist = l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                                l_dist += l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                            l_dist = l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:                                l_dist += l1distance_cuda(
FastGeodis/fastgeodis_cuda.cu:void geodesic_frontback_pass_cuda(
FastGeodis/fastgeodis_cuda.cu:    // copy local distances to GPU __constant__ memory
FastGeodis/fastgeodis_cuda.cu:    cudaMemcpyToSymbol(local_dist3d, local_dist, sizeof(float) * 3 * 3);
FastGeodis/fastgeodis_cuda.cu:torch::Tensor generalised_geodesic3d_cuda(
FastGeodis/fastgeodis_cuda.cu:    // std::cout << "Running with CUDA Device: " << device << std::endl;
FastGeodis/fastgeodis_cuda.cu:    c10::cuda::CUDAGuard device_guard(device);
FastGeodis/fastgeodis_cuda.cu:        geodesic_frontback_pass_cuda(image_local, distance, spacing, l_grad, l_eucl);
FastGeodis/fastgeodis_cuda.cu:        geodesic_frontback_pass_cuda(
FastGeodis/fastgeodis_cuda.cu:        geodesic_frontback_pass_cuda(
FastGeodis/fastgeodis.cpp:        #ifdef WITH_CUDA
FastGeodis/fastgeodis.cpp:            std::cout << "Compiled with CUDA support" << std::endl;
FastGeodis/fastgeodis.cpp:            std::cout << "Not compiled with CUDA support" << std::endl;
FastGeodis/fastgeodis.cpp:    if (image.is_cuda()) 
FastGeodis/fastgeodis.cpp:    #ifdef WITH_CUDA
FastGeodis/fastgeodis.cpp:        if (!torch::cuda::is_available())
FastGeodis/fastgeodis.cpp:                "cuda.is_available() returned false, please check if the library was compiled successfully with CUDA support");
FastGeodis/fastgeodis.cpp:        check_cuda(mask);
FastGeodis/fastgeodis.cpp:        return generalised_geodesic2d_cuda(image, mask, v, l_grad, l_eucl, iterations);
FastGeodis/fastgeodis.cpp:        AT_ERROR("Not compiled with CUDA support.");
FastGeodis/fastgeodis.cpp:        #ifdef WITH_CUDA
FastGeodis/fastgeodis.cpp:            std::cout << "Compiled with CUDA support" << std::endl;
FastGeodis/fastgeodis.cpp:            std::cout << "Not compiled with CUDA support" << std::endl;
FastGeodis/fastgeodis.cpp:    if (image.is_cuda()) 
FastGeodis/fastgeodis.cpp:    #ifdef WITH_CUDA
FastGeodis/fastgeodis.cpp:        if (!torch::cuda::is_available())
FastGeodis/fastgeodis.cpp:                "cuda.is_available() returned false, please check if the library was compiled successfully with CUDA support");
FastGeodis/fastgeodis.cpp:        check_cuda(mask);
FastGeodis/fastgeodis.cpp:        return generalised_geodesic3d_cuda(image, mask, spacing, v, l_grad, l_eucl, iterations);
FastGeodis/fastgeodis.cpp:        AT_ERROR("Not compiled with CUDA support.");
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location
FastGeodis/__init__.py:    The function expects input as torch.Tensor, which can be run on CPU or GPU depending on Tensor's device location

```
