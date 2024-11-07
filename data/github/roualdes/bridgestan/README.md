# https://github.com/roualdes/bridgestan

```console
python/test/test_stanmodel.py:    assert "STAN_OPENCL" in b.model_info()
R/tests/testthat/test_model.R:    expect_true(grepl("STAN_OPENCL", simple$model_info(), fixed = TRUE))
docs/getting-started.rst:Additional flags, such as those for MPI and OpenCL, are covered in the
Makefile:# set flags for stanc compiler (math calls MIGHT? set STAN_OPENCL)
Makefile:ifdef STAN_OPENCL
Makefile:	override STANCFLAGS += --use-opencl
Makefile:	STAN_FLAG_OPENCL=_opencl
Makefile:	STAN_FLAG_OPENCL=
Makefile:STAN_FLAGS=$(STAN_FLAG_THREADS)$(STAN_FLAG_OPENCL)$(STAN_FLAG_HESS)
src/model_rng.cpp:#ifdef STAN_OPENCL
src/model_rng.cpp:  info << "\tSTAN_OPENCL=true" << std::endl;
src/model_rng.cpp:  info << "\tSTAN_OPENCL=false" << std::endl;

```
