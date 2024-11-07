# https://github.com/mathLab/PINA

```console
docs/source/_rst/tutorials/tutorial8/tutorial.rst:    GPU available: True (cuda), used: False
docs/source/_rst/tutorials/tutorial8/tutorial.rst:    /u/a/aivagnes/anaconda3/lib/python3.8/site-packages/pytorch_lightning/trainer/setup.py:187: GPU available but not used. You can set it by doing `Trainer(accelerator='gpu')`.
docs/source/_rst/tutorials/tutorial8/tutorial.rst:    /u/a/aivagnes/anaconda3/lib/python3.8/site-packages/torch/cuda/__init__.py:152: UserWarning:
docs/source/_rst/tutorials/tutorial8/tutorial.rst:        Found GPU0 Quadro K600 which is of cuda capability 3.0.
docs/source/_rst/tutorials/tutorial8/tutorial.rst:        PyTorch no longer supports this GPU because it is too old.
docs/source/_rst/tutorials/tutorial8/tutorial.rst:        The minimum cuda capability supported by this library is 3.7.
docs/source/_rst/tutorials/tutorial8/tutorial.rst:      warnings.warn(old_gpu_warn % (d, name, major, minor, min_arch // 10, min_arch % 10))
docs/source/_rst/tutorials/tutorial10/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial1/tutorial.rst:3. GPU training and speed benchmarking
docs/source/_rst/tutorials/tutorial9/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: True
docs/source/_rst/tutorials/tutorial11/tutorial.rst:4. `GPU <https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html#:~:text=What%20does%20GPU%20stand%20for,video%20editing%2C%20and%20gaming%20applications>`__ or `MPS <https://developer.apple.com/metal/pytorch/>`__
docs/source/_rst/tutorials/tutorial11/tutorial.rst:-  ``accelerator = {'gpu', 'cpu', 'hpu', 'mps', 'cpu', 'ipu'}`` sets the
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:as you can see, even if in the used system ``GPU`` is available, it is
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial11/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial13/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial13/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial13/tutorial.rst:    GPU available: True (mps), used: False
docs/source/_rst/tutorials/tutorial5/tutorial.rst:    GPU available: False, used: False
docs/source/_rst/tutorials/tutorial5/tutorial.rst:    GPU available: False, used: False
docs/source/_rst/tutorials/tutorial5/tutorial.rst:``FeedForward`` network. We suggest to use GPU or TPU for a speed up in
joss/paper.bib:  title = {{NVIDIA Modulus}},
joss/paper.bib:  howpublished = "\url{https://github.com/NVIDIA/modulus}",
joss/paper.bib:  title={TensorDiffEq: Scalable Multi-GPU Forward and Inverse Solvers for Physics Informed Neural Networks},
joss/paper.bib:@inproceedings{hennigh2021nvidia,
joss/paper.bib:  title={NVIDIA SimNet™: An AI-accelerated multi-physics simulation framework},
joss/paper.md:We here mention some PyTorch-based libraries, \verb+NeuroDiffEq+ [@chen2020neurodiffeq], \verb+IDRLNet+ [@peng2021idrlnet], NVIDIA \verb+Modulus+ [@modulussym], and some TensorFlow-based libraries, such as \verb+DeepXDE+ [@lu2021deepxde], \verb+TensorDiffEq+ [@mcclenny2021tensordiffeq], \verb+SciANN+ [@haghighat2021sciann] (which is both TensorFlow and Keras-based), \verb+PyDEns+ [@koryagin2019pydens], \verb+Elvet+ [@araz2021elvet], \verb+NVIDIA SimNet+ [@hennigh2021nvidia].
tutorials/tutorial1/tutorial.py:# 3. GPU training and speed benchmarking
tutorials/tutorial11/tutorial.py:# 4. [GPU](https://www.intel.com/content/www/us/en/products/docs/processors/what-is-a-gpu.html#:~:text=What%20does%20GPU%20stand%20for,video%20editing%2C%20and%20gaming%20applications) or [MPS](https://developer.apple.com/metal/pytorch/)
tutorials/tutorial11/tutorial.py:# * `accelerator = {'gpu', 'cpu', 'hpu', 'mps', 'cpu', 'ipu'}` sets the accelerator to a specific one
tutorials/tutorial11/tutorial.py:# as you can see, even if in the used system `GPU` is available, it is not used since we set `accelerator='cpu'`.
tutorials/tutorial5/tutorial.py:# We can clearly see that the final loss is lower. Let's see in testing.. Notice that the number of parameters is way higher than a `FeedForward` network. We suggest to use GPU or TPU for a speed up in training, when many data samples are used.
tests/test_solvers/test_competitive_pinn.py:# TODO, fix GitHub actions to run also on GPU
tests/test_solvers/test_competitive_pinn.py:# def test_train_gpu():
tests/test_solvers/test_competitive_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_competitive_pinn.py:# def test_train_gpu(): #TODO fix ASAP
tests/test_solvers/test_competitive_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_competitive_pinn.py:# if torch.cuda.is_available():
tests/test_solvers/test_competitive_pinn.py:#     # def test_gpu_train():
tests/test_solvers/test_competitive_pinn.py:#     #     pinn = PINN(problem, model, batch_size=20, device='cuda')
tests/test_solvers/test_competitive_pinn.py:#     def test_gpu_train_nobatch():
tests/test_solvers/test_competitive_pinn.py:#         pinn = PINN(problem, model, batch_size=None, device='cuda')
tests/test_solvers/test_sapinn.py:# TODO, fix GitHub actions to run also on GPU
tests/test_solvers/test_sapinn.py:# def test_train_gpu():
tests/test_solvers/test_sapinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_sapinn.py:# def test_train_gpu(): #TODO fix ASAP
tests/test_solvers/test_sapinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_sapinn.py:# if torch.cuda.is_available():
tests/test_solvers/test_sapinn.py:#     # def test_gpu_train():
tests/test_solvers/test_sapinn.py:#     #     pinn = PINN(problem, model, batch_size=20, device='cuda')
tests/test_solvers/test_sapinn.py:#     def test_gpu_train_nobatch():
tests/test_solvers/test_sapinn.py:#         pinn = PINN(problem, model, batch_size=None, device='cuda')
tests/test_solvers/test_pinn.py:# TODO, fix GitHub actions to run also on GPU
tests/test_solvers/test_pinn.py:# def test_train_gpu():
tests/test_solvers/test_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_pinn.py:# def test_train_gpu(): #TODO fix ASAP
tests/test_solvers/test_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_pinn.py:# if torch.cuda.is_available():
tests/test_solvers/test_pinn.py:#     # def test_gpu_train():
tests/test_solvers/test_pinn.py:#     #     pinn = PINN(problem, model, batch_size=20, device='cuda')
tests/test_solvers/test_pinn.py:#     def test_gpu_train_nobatch():
tests/test_solvers/test_pinn.py:#         pinn = PINN(problem, model, batch_size=None, device='cuda')
tests/test_solvers/test_gpinn.py:# TODO, fix GitHub actions to run also on GPU
tests/test_solvers/test_gpinn.py:# def test_train_gpu():
tests/test_solvers/test_gpinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_gpinn.py:# def test_train_gpu(): #TODO fix ASAP
tests/test_solvers/test_gpinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_gpinn.py:# if torch.cuda.is_available():
tests/test_solvers/test_gpinn.py:#     # def test_gpu_train():
tests/test_solvers/test_gpinn.py:#     #     pinn = GPINN(problem, model, batch_size=20, device='cuda')
tests/test_solvers/test_gpinn.py:#     def test_gpu_train_nobatch():
tests/test_solvers/test_gpinn.py:#         pinn = GPINN(problem, model, batch_size=None, device='cuda')
tests/test_solvers/test_rba_pinn.py:# TODO, fix GitHub actions to run also on GPU
tests/test_solvers/test_rba_pinn.py:# def test_train_gpu():
tests/test_solvers/test_rba_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_rba_pinn.py:# def test_train_gpu(): #TODO fix ASAP
tests/test_solvers/test_rba_pinn.py:#     trainer = Trainer(solver=pinn, kwargs={'max_epochs' : 5, 'accelerator':'gpu'})
tests/test_solvers/test_rba_pinn.py:# if torch.cuda.is_available():
tests/test_solvers/test_rba_pinn.py:#     # def test_gpu_train():
tests/test_solvers/test_rba_pinn.py:#     #     pinn = PINN(problem, model, batch_size=20, device='cuda')
tests/test_solvers/test_rba_pinn.py:#     def test_gpu_train_nobatch():
tests/test_solvers/test_rba_pinn.py:#         pinn = PINN(problem, model, batch_size=None, device='cuda')
README.md:trainer = Trainer(pinn, max_epochs=1000, accelerator='gpu', enable_model_summary=False, batch_size=8)
pina/operators.py:These operators are implemented to work on different accellerators: CPU, GPU, TPU or MPS.
pina/solvers/pinns/causalpinn.py:        # sort the time tensors (this is very bad for GPU)
pina/label_tensor.py:    def cuda(self, *args, **kwargs):
pina/label_tensor.py:        Send Tensor to cuda. For more details, see :meth:`torch.Tensor.cuda`.
pina/label_tensor.py:        tmp = super().cuda(*args, **kwargs)

```
