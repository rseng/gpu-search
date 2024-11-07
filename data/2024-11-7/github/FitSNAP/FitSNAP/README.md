# https://github.com/FitSNAP/FitSNAP

```console
docs/source/Pytorch.rst:GPU Acceleration
docs/source/Pytorch.rst:FitSNAP supports GPU acceleration via PyTorch. With small batch sizes, however, most of the benefit 
docs/source/Pytorch.rst:of GPU parallelization comes from evaluating the NN model and calculating gradients. You will not see 
docs/source/Pytorch.rst:a large benefit of GPUs using a small batch size unless you have a large NN model (e.g. > 1 million 
docs/source/Pytorch.rst:parameters). If you have a small model, you will see a speedup on GPUs using a large enough batch 
docs/PARALLEL.md:their respective linear equations with threading or GPUs depending on the choice of solver.
fitsnap3lib/lib/neural_networks/write.py:        # TODO: Make GPU option, need to implement into ML-IAP.
fitsnap3lib/lib/neural_networks/write.py:        # TODO: Make GPU option, need to implement into ML-IAP.
fitsnap3lib/solvers/network.py:            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fitsnap3lib/solvers/network.py:                    prevent device mismatch when training on GPU.
fitsnap3lib/solvers/pytorch.py:            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fitsnap3lib/solvers/pytorch.py:                    prevent device mismatch when training on GPU.

```
