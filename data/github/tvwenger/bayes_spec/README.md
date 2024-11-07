# https://github.com/tvwenger/bayes_spec

```console
environment-cuda.yml:  - nvidia
environment-cuda.yml:  - nvidia::cuda-toolkit
environment-cuda.yml:  - jaxlib[cuda]
README.md:# or, if you would like to use CUDA (nvidia GPU) samplers:
README.md:# conda env create -f environment-cuda.yml
README.md:`bayes_spec` can also use MCMC to sample the posterior distribution. MCMC sampling tends to be much slower but also more accurate. Draw posterior samples using MCMC via `model.sample()`. Since `bayes_spec` uses `pymc` for sampling, several `pymc` samplers are available, including GPU samplers (see ["other samplers" example notebook](https://bayes-spec.readthedocs.io/en/stable/notebooks/other_samplers.html)).

```
