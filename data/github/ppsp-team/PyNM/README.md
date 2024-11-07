# https://github.com/ppsp-team/PyNM

```console
README.md:The approximate model implements a Stochastic Variational Gaussian Process (SVGP) model using [GPytorch](https://gpytorch.ai/), with a kernel closely matching the one in the exact model. SVGP is a deep learning technique that needs to be trained on minibatches for a set number of epochs, this can be tuned with the parameters `batch_size` and `num_epoch`. The model speeds up computation by using a subset of the data as inducing points, this can be controlled with the parameter `n_inducing` that defines how many points to use. See [documentation](https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html) for an overview.
pynm/pynm.py:        https://docs.gpytorch.ai/en/v1.1.1/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html#Creating-a-SVGP-Model.
pynm/models/approx.py:        if torch.cuda.is_available():
pynm/models/approx.py:            self.model = self.model.cuda()
pynm/models/approx.py:            self.likelihood = self.likelihood.cuda()

```
