# https://github.com/jmschrei/pomegranate

```console
docs/faq.rst:**Does pomegranate support GPUs?**
docs/faq.rst:Yes! Again, because pomegranate v1.0.0 is written in PyTorch, every algorithm has GPU support. The speed increase scales with the complexity of the algorithm, with simple probability distributions having approximately a ~2-3x speedup whereas the forward-backward algorithm for hidden Markov models can be up to ~5-10x faster by using a GPU.
docs/whats_new.rst:	- Fixed an issue with categorical distributions being used on the GPU
docs/whats_new.rst:	- GPU support has been added for all models and methods
docs/whats_new.rst:	- GPU acceleration should be fixed
docs/whats_new.rst:	- The `enable_gpu` call has been moved to the bottom of the GPU checking code and so should not crash anymore.
docs/whats_new.rst:	- k-means has been changed from using iterative computation to using the alternate formulation of euclidean distance, from ||a - b||^{2} to using ||a||^{2} + ||b||^{2} - 2||a \cdot b||. This allows for the centroid norms to be cached, significantly speeding up computation, and for dgemm to be used to solve the matrix matrix multiplication. Initial attempts to add in GPU support appeared unsuccessful, but in theory it should be something that can be added in.
docs/whats_new.rst:	- Multivariate Gaussian Distributions can now use GPUs for both log probability and summarization calculations, speeding up both tasks ~4x for any models that use them. This is added in through CuPy.
docs/index.rst:In addition to a variety of probability distributions and models, pomegranate has a variety of built-in features that are implemented for all of the models. These include different training strategies such as semi-supervised learning, learning with missing values, and mini-batch learning. It also includes support for massive data supports with out-of-core learning, multi-threaded parallelism, and GPU support. 
docs/index.rst:   tutorials/C_Feature_Tutorial_1_GPU_Usage.ipynb
docs/install.rst:Because pomegranate recently moved to a PyTorch backend, the most complicated installation step now is likely installing that and its CUDA dependencies. Please see the PyTorch documentation for help installing those.
README.md:- <b>Features</b>: PyTorch has many features, such as serialization, mixed precision, and GPU support, that can now be directly used in pomegranate without additional work on my end. 
README.md:- All models now have GPU support
README.md:### GPU Support
README.md:All distributions and methods in pomegranate now have GPU support. Because each distribution is a `torch.nn.Module` object, the use is identical to other code written in PyTorch. This means that both the model and the data have to be moved to the GPU by the user. For instance:
README.md:# Will execute on a GPU
README.md:>>> d = Exponential().cuda().fit(X.cuda())
README.md:tensor([1.8627, 1.3132, 1.7187, 1.4957], device='cuda:0')
README.md:Likewise, all models are distributions, and so can be used on the GPU similarly. When a model is moved to the GPU, all of the models associated with it (e.g. distributions) are also moved to the GPU.
README.md:>>> X = torch.exp(torch.randn(50, 4)).cuda()
README.md:>>> model = GeneralMixtureModel([Exponential(), Exponential()]).cuda()
README.md:tensor([0.9141, 1.0835, 2.7503, 2.2475], device='cuda:0')
README.md:tensor([1.9902, 2.3871, 0.8984, 1.2215], device='cuda:0')
README.md:>>> with torch.autocast('cuda', dtype=torch.bfloat16):
README.md:>>> X = torch.exp(torch.randn(50, 4)).cuda()
README.md:>>> model.cuda()
README.md:In PyTorch v2.0.0, `torch.compile` was introduced as a flexible wrapper around tools that would fuse operations together, use CUDA graphs, and generally try to remove I/O bottlenecks in GPU execution. Because these bottlenecks can be extremely significant in the small-to-medium sized data settings many pomegranate users are faced with, `torch.compile` seems like it will be extremely valuable. Rather than targeting entire models, which mostly just compiles the `forward` method, you should compile individual methods from your objects.
README.md:>>> d = Exponential(mu).cuda()

```
