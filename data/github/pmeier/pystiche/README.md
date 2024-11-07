# https://github.com/pmeier/pystiche

```console
docs/source/getting_started/installation.rst:install the PyTorch distributions precompiled for the latest CUDA release. If you use
docs/source/getting_started/installation.rst:another version or don't have a CUDA-capable GPU, we encourage you to try
docs/source/getting_started/installation.rst:  While ``pystiche`` is designed to be fully functional without a GPU, most tasks
docs/source/references.bib:  abstract  = {We trained a large, deep convolutional neural network to classify the 1.2 million high-resolution images in the ImageNet ILSVRC-2010 contest into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 37.5% and 17.0% which is considerably better than the previous state-of-the-art. The neural network, which has 60 million parameters and 650,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers,and three fully-connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of the convolution operation. To reduce overfitting in the fully-connected layers we employed a recently-developed regularization method called “dropout” that proved to be very effective. We also entered a variant of this model in the ILSVRC-2012 competition and achieved a winning top-5 test error rate of 15.3%,compared to 26.2% achieved by the second-best entry.},
docs/source/references.bib:  url       = {http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf},
docs/source/conf.py:    def show_cuda_memory(func):
docs/source/conf.py:        torch.cuda.reset_peak_memory_stats()
docs/source/conf.py:        stats = torch.cuda.memory_stats()
docs/source/conf.py:    if plot_gallery and not torch.cuda.is_available():
docs/source/conf.py:            "The galleries will be built, but CUDA is not available. "
docs/source/conf.py:        "show_memory": show_cuda_memory if torch.cuda.is_available() else True,
tests/utils.py:    "skip_if_cuda_not_available",
tests/utils.py:skip_if_cuda_not_available = pytest.mark.skipif(
tests/utils.py:    not torch.cuda.is_available(), reason="CUDA is not available."
tests/integration/core/test_objects.py:from tests.utils import skip_if_cuda_not_available
tests/integration/core/test_objects.py:    @skip_if_cuda_not_available
tests/integration/core/test_objects.py:        key2 = pystiche.TensorKey(x.cuda())
tests/integration/misc/test_misc.py:        desired = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tests/conftest.py:            "@slow_if_cuda_not_available if CUDA is not available."
tests/conftest.py:        keyword="slow_if_cuda_not_available",
tests/conftest.py:                "slow_if_cuda_not_available" in item.keywords
tests/conftest.py:                and not torch.cuda.is_available()
tests/conftest.py:        reason="Test is slow since CUDA is not available and --skip-slow was given.",
.cirun.yml:  - name: docs-build-gpu
.cirun.yml:    # Amazon AMI: Deep Learning AMI GPU CUDA 11.2.1 (Ubuntu 20.04) 20210625
.cirun.yml:      - gpu
pytest.ini:  slow_if_cuda_not_available
scripts/README.md:1. Installing PyTorch distributionswith CUDA support exceeds their memory limit. Thus, 
examples/README.rst:  Although a GPU is not a requirement, it is strongly advised to run these examples
examples/README.rst:  with one. If you don't have access to a GPU, the execution time of the examples might
examples/README.rst:  of each example is measured using a GPU.
examples/beginner/example_nst_without_pystiche.py:device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pystiche/_cli.py:        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
pystiche/_cli.py:            "If available, defaults to 'cuda' and falls back to 'cpu' otherwise."
pystiche/misc.py:            ``None`` selects CUDA if available and otherwise CPU. Defaults to ``None``.
pystiche/misc.py:        device = "cuda" if torch.cuda.is_available() else "cpu"

```
