# https://github.com/ropensci/stantargets

```console
inst/WORDLIST:OpenCL
R/tar_stan_mcmc_rep_draws.R:  opencl_ids = NULL,
R/tar_stan_mcmc_rep_draws.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc_rep_diagnostics.R:  opencl_ids = NULL,
R/tar_stan_mcmc_rep_diagnostics.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc.R:  opencl_ids = NULL,
R/tar_stan_mcmc.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc.R:  opencl_ids,
R/tar_stan_mcmc.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc_rep_summary.R:  opencl_ids = NULL,
R/tar_stan_mcmc_rep_summary.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc_rep.R:  opencl_ids = NULL,
R/tar_stan_mcmc_rep.R:    opencl_ids = opencl_ids,
R/tar_stan_mcmc_rep.R:  opencl_ids,
R/tar_stan_mcmc_rep.R:      opencl_ids = opencl_ids,
R/tar_stan_mcmc_rep.R:  opencl_ids,
R/tar_stan_mcmc_rep.R:    opencl_ids = opencl_ids,
man/tar_stan_mcmc_rep_summary.Rd:  opencl_ids = NULL,
man/tar_stan_mcmc_rep_summary.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_summary.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_rep_summary.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_rep_summary.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_gq_rep_summary.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_vb.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_run.Rd:  opencl_ids,
man/tar_stan_mcmc_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_run.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_run.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_run.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_mle_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_gq_rep_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_gq.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_vb_rep_draws.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_vb_rep_summary.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_run.Rd:  opencl_ids,
man/tar_stan_mcmc_rep_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_run.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_rep_run.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_rep_run.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_vb_rep_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc.Rd:  opencl_ids = NULL,
man/tar_stan_mcmc.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_vb_rep.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_gq_rep.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_gq_rep_draws.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_vb_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep.Rd:  opencl_ids = NULL,
man/tar_stan_mcmc_rep.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_rep.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_rep.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_mle_rep_draws.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_gq_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mle_rep_summary.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mle.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_diagnostics.Rd:  opencl_ids = NULL,
man/tar_stan_mcmc_rep_diagnostics.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_diagnostics.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_rep_diagnostics.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_rep_diagnostics.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_compile.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mle_rep_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_compile_run.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_draws.Rd:  opencl_ids = NULL,
man/tar_stan_mcmc_rep_draws.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would
man/tar_stan_mcmc_rep_draws.Rd:\item{opencl_ids}{(integer vector of length 2) The platform and
man/tar_stan_mcmc_rep_draws.Rd:device IDs of the OpenCL device to use for fitting. The model must
man/tar_stan_mcmc_rep_draws.Rd:be compiled with \code{cpp_options = list(stan_opencl = TRUE)} for this
man/tar_stan_mle_rep.Rd:model (\code{STAN_THREADS}, \code{STAN_MPI}, \code{STAN_OPENCL}, etc.). Anything you would

```
