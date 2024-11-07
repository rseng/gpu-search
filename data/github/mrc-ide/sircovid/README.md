# https://github.com/mrc-ide/sircovid

```console
inst/dust/lancelot.cpp:// These exist to support the model on the gpu, as in C++14 std::min
inst/dust/basic.cpp:// These exist to support the model on the gpu, as in C++14 std::min
inst/odin/lancelot.R:## on a GPU, and with fast math, the sum can include sufficient
buildkite/pipeline.yml:  - label: ":allthethings: Build cuda image"
buildkite/pipeline.yml:    command: docker/build_cuda
R/gpu.R:##' Create a model for the GPU. This requires a working nvcc toolchain
R/gpu.R:##' and GPU device to work properly. Note that this will cache the
R/gpu.R:##' compilation within a session, so if you want to change GPU options
R/gpu.R:##' @title Create GPU model
R/gpu.R:##'   and from there into either odin's options or the gpu options.
R/gpu.R:##'   typically what you want on the GPU.
R/gpu.R:##' @param gpu The argument passed to [odin.dust::odin_dust_] as
R/gpu.R:##'   `gpu`. The default here is `TRUE`, but you may want to pass the
R/gpu.R:##'   results of [dust::dust_cuda_options] in order to control
R/gpu.R:##'   on the GPU available to this computer.
R/gpu.R:compile_gpu <- function(model = "lancelot", ..., real_type = "float",
R/gpu.R:                        gpu = TRUE) {
R/gpu.R:  odin.dust::odin_dust_(path, real_type = real_type, gpu = gpu, ...)
R/cpp11.R:dust_basic_gpu_info <- function() {
R/cpp11.R:  .Call(`_sircovid_dust_basic_gpu_info`)
R/cpp11.R:dust_cpu_basic_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_sircovid_dust_cpu_basic_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/cpp11.R:dust_lancelot_gpu_info <- function() {
R/cpp11.R:  .Call(`_sircovid_dust_lancelot_gpu_info`)
R/cpp11.R:dust_cpu_lancelot_alloc <- function(r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control) {
R/cpp11.R:  .Call(`_sircovid_dust_cpu_lancelot_alloc`, r_pars, pars_multi, r_time, r_n_particles, n_threads, r_seed, deterministic, gpu_config, ode_control)
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_basic_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_basic_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
R/dust.R:    gpu_config_ = NULL,
R/dust.R:                          gpu_config = NULL, ode_control = NULL) {
R/dust.R:      if (is.null(gpu_config)) {
R/dust.R:          stop("GPU support not enabled for this object")
R/dust.R:                        n_threads, seed, deterministic, gpu_config, ode_control)
R/dust.R:      private$gpu_config_ <- res[[4L]]
R/dust.R:    has_gpu_support = function(fake_gpu = FALSE) {
R/dust.R:      if (fake_gpu) {
R/dust.R:        dust_cpu_lancelot_capabilities()[["gpu"]]
R/dust.R:    uses_gpu = function(fake_gpu = FALSE) {
R/dust.R:      real_gpu <- private$gpu_config_$real_gpu
R/dust.R:      !is.null(real_gpu) && (fake_gpu || real_gpu)
R/dust.R:    gpu_info = function() {
R/dust.R:      ret <- dust_lancelot_gpu_info()
R/dust.R:      if (ret$has_cuda && exists("private", parent, inherits = FALSE)) {
R/dust.R:        ret$config <- private$gpu_config_
tests/testthat/test-gpu.R:context("gpu")
tests/testthat/test-gpu.R:test_that("can generate GPU interface and pass arguments", {
tests/testthat/test-gpu.R:  mockery::stub(compile_gpu, "odin.dust::odin_dust_", mock_odin_dust)
tests/testthat/test-gpu.R:    compile_gpu(),
tests/testthat/test-gpu.R:    list(sircovid_file("odin/lancelot.R"), real_type = "float", gpu = TRUE))
tests/testthat/test-gpu.R:  compile_gpu(real_type = "double", rewrite_dims = TRUE,
tests/testthat/test-gpu.R:              gpu = FALSE, gpu_generate = TRUE)
tests/testthat/test-gpu.R:    list(sircovid_file("odin/lancelot.R"), real_type = "double", gpu = FALSE,
tests/testthat/test-gpu.R:         rewrite_dims = TRUE, gpu_generate = TRUE))
tests/testthat/test-gpu.R:test_that("can run the gpu model on the cpu", {
tests/testthat/test-gpu.R:  gen <- compile_gpu(gpu = FALSE, gpu_generate = TRUE, verbose = FALSE)
tests/testthat/test-gpu.R:  mod_gpu <- gen$new(p, 0, 5, seed = 1L, gpu_config = 0)
tests/testthat/test-gpu.R:  expect_equal(mod_gpu$info(), mod_cpu$info())
tests/testthat/test-gpu.R:  mod_gpu$update_state(state = initial)
tests/testthat/test-gpu.R:  mod_gpu$set_index(index)
tests/testthat/test-gpu.R:  res_gpu <- mod_gpu$run(end)
tests/testthat/test-gpu.R:  expect_equal(res_gpu, res_cpu)
tests/testthat/test-gpu.R:test_that("Can run the gpu compare on the cpu", {
tests/testthat/test-gpu.R:  gen <- compile_gpu(gpu = FALSE, gpu_generate = TRUE, verbose = FALSE)
tests/testthat/test-gpu.R:  mod_gpu <- gen$new(pars, 0, np, seed = 1L, gpu_config = 0)
tests/testthat/test-gpu.R:  mod_gpu$update_state(state = initial)
tests/testthat/test-gpu.R:  mod_gpu$set_data(dust::dust_data(data, "time_end"))
tests/testthat/test-gpu.R:  y_gpu <- mod_gpu$run(data$time_end[[i]])
tests/testthat/test-gpu.R:  expect_equal(mod_cpu$compare_data(), mod_gpu$compare_data())
docker/build_cuda:SIRCOVID_CUDA="${PACKAGE_ORG}/${PACKAGE_NAME}-cuda:${GIT_SHA}"
docker/build_cuda:       --tag $SIRCOVID_CUDA \
docker/build_cuda:       -f $PACKAGE_ROOT/docker/Dockerfile.cuda \
docker/build_cuda:docker run -it --rm -v $PWD:/src:ro $SIRCOVID_CUDA /compile_cuda_model
docker/compile_cuda_model:message("Checking CUDA configuration")
docker/compile_cuda_model:dust::dust_cuda_configuration(quiet = FALSE)
docker/compile_cuda_model:gen <- sircovid::compile_gpu(gpu = TRUE, verbose = TRUE)
docker/Dockerfile.cuda:        nvidia-cuda-toolkit \
docker/Dockerfile.cuda:RUN Rscript -e 'dust:::cuda_install_cub(NULL)'
docker/Dockerfile.cuda:COPY docker/compile_cuda_model /
NEWS.md:* New function `sircovid::compile_gpu` which compiles sircovid to run on a GPU (#237)
NAMESPACE:export(compile_gpu)
man/compile_gpu.Rd:% Please edit documentation in R/gpu.R
man/compile_gpu.Rd:\name{compile_gpu}
man/compile_gpu.Rd:\alias{compile_gpu}
man/compile_gpu.Rd:\title{Create GPU model}
man/compile_gpu.Rd:compile_gpu(model = "lancelot", ..., real_type = "float", gpu = TRUE)
man/compile_gpu.Rd:and from there into either odin's options or the gpu options.}
man/compile_gpu.Rd:typically what you want on the GPU.}
man/compile_gpu.Rd:\item{gpu}{The argument passed to \link[odin.dust:odin_dust]{odin.dust::odin_dust_} as
man/compile_gpu.Rd:\code{gpu}. The default here is \code{TRUE}, but you may want to pass the
man/compile_gpu.Rd:results of \link[dust:dust_cuda_options]{dust::dust_cuda_options} in order to control
man/compile_gpu.Rd:on the GPU available to this computer.
man/compile_gpu.Rd:Create a model for the GPU. This requires a working nvcc toolchain
man/compile_gpu.Rd:and GPU device to work properly. Note that this will cache the
man/compile_gpu.Rd:compilation within a session, so if you want to change GPU options
src/cpp11.cpp:cpp11::sexp dust_basic_gpu_info();
src/cpp11.cpp:extern "C" SEXP _sircovid_dust_basic_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_basic_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_basic_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _sircovid_dust_cpu_basic_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_basic_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:cpp11::sexp dust_lancelot_gpu_info();
src/cpp11.cpp:extern "C" SEXP _sircovid_dust_lancelot_gpu_info() {
src/cpp11.cpp:    return cpp11::as_sexp(dust_lancelot_gpu_info());
src/cpp11.cpp:SEXP dust_cpu_lancelot_alloc(cpp11::list r_pars, bool pars_multi, cpp11::sexp r_time, cpp11::sexp r_n_particles, int n_threads, cpp11::sexp r_seed, bool deterministic, cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/cpp11.cpp:extern "C" SEXP _sircovid_dust_cpu_lancelot_alloc(SEXP r_pars, SEXP pars_multi, SEXP r_time, SEXP r_n_particles, SEXP n_threads, SEXP r_seed, SEXP deterministic, SEXP gpu_config, SEXP ode_control) {
src/cpp11.cpp:    return cpp11::as_sexp(dust_cpu_lancelot_alloc(cpp11::as_cpp<cpp11::decay_t<cpp11::list>>(r_pars), cpp11::as_cpp<cpp11::decay_t<bool>>(pars_multi), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_time), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_n_particles), cpp11::as_cpp<cpp11::decay_t<int>>(n_threads), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(r_seed), cpp11::as_cpp<cpp11::decay_t<bool>>(deterministic), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(gpu_config), cpp11::as_cpp<cpp11::decay_t<cpp11::sexp>>(ode_control)));
src/cpp11.cpp:    {"_sircovid_dust_basic_gpu_info",                       (DL_FUNC) &_sircovid_dust_basic_gpu_info,                       0},
src/cpp11.cpp:    {"_sircovid_dust_lancelot_gpu_info",                    (DL_FUNC) &_sircovid_dust_lancelot_gpu_info,                    0},
src/lancelot.cpp:cpp11::sexp dust_lancelot_gpu_info();
src/lancelot.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/lancelot.cpp:// These exist to support the model on the gpu, as in C++14 std::min
src/lancelot.cpp:cpp11::sexp dust_lancelot_gpu_info() {
src/lancelot.cpp:  return dust::gpu::r::gpu_info();
src/lancelot.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/lancelot.cpp:                                        gpu_config, ode_control);
src/basic.cpp:cpp11::sexp dust_basic_gpu_info();
src/basic.cpp:                         cpp11::sexp gpu_config, cpp11::sexp ode_control);
src/basic.cpp:// These exist to support the model on the gpu, as in C++14 std::min
src/basic.cpp:cpp11::sexp dust_basic_gpu_info() {
src/basic.cpp:  return dust::gpu::r::gpu_info();
src/basic.cpp:                             cpp11::sexp gpu_config, cpp11::sexp ode_control) {
src/basic.cpp:                                        gpu_config, ode_control);

```
