# https://github.com/ICRAR/ProFit

```console
R/profitMakeModel.R:                           openclenv=NULL, omp_threads=NULL, plot=FALSE, ...) {
R/profitMakeModel.R:	if (!is.null(openclenv)) {
R/profitMakeModel.R:		if (class(openclenv) == 'externalptr') {
R/profitMakeModel.R:			model[['openclenv']] = openclenv
R/profitMakeModel.R:		else if (openclenv == 'get') {
R/profitMakeModel.R:			model[['openclenv']] = profitOpenCLEnv()
R/profitBenchmark.R:.profitGetOpenCLEnvRow <- function(name="",env_i=NA,
R/profitBenchmark.R:  openclenv = NULL
R/profitBenchmark.R:  if(identical(result$name[best],"opencl"))
R/profitBenchmark.R:    openclenv = result[[best,paste0("env_",precision)]]
R/profitBenchmark.R:    openclenv=openclenv,
R/profitBenchmark.R:                            openclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
R/profitBenchmark.R:  stopifnot(is.data.frame(openclenvs))
R/profitBenchmark.R:  hasopenclenvs = nrow(openclenvs) > 0
R/profitBenchmark.R:  if(!hasopenclenvs)
R/profitBenchmark.R:    openclenvs = NULL
R/profitBenchmark.R:    oclmethods = which(startsWith(methods,"opencl"))
R/profitBenchmark.R:    # Rename according to the subtype e.g. if opencl-local is supported
R/profitBenchmark.R:    if(startsWith(method,"opencl"))
R/profitBenchmark.R:      stopifnot(nrow(openclenvs) > 0)
R/profitBenchmark.R:      openclenvs$name = method
R/profitBenchmark.R:      bench = rbind(bench,openclenvs)
R/profitBenchmark.R:      bench = rbind(bench,.profitGetOpenCLEnvRow(name=method))
R/profitBenchmark.R:        openclenv = bench[[paste0("env_",prec)]][[methodi]]
R/profitBenchmark.R:        if(identical(openclenv,new("externalptr")))
R/profitBenchmark.R:          if(startsWith(method,"opencl"))
R/profitBenchmark.R:            stop(paste0("Error! OpenCL method='",method,"', env='",bench$env_name[[methodi]],
R/profitBenchmark.R:                        "', has null openclptr. Did you call profitGetOpenCLEnvs(make=TRUE)?"))
R/profitBenchmark.R:          openclenv=NULL
R/profitBenchmark.R:                                          omp_threads=omp_threads, openclenv=openclenv)
R/profitBenchmark.R:                                       openclenv = openclenv, omp_threads = omp_threads)$z
R/profitLikeModel.R:    openclenv=Data$openclenv
R/profitLikeModel.R:    if(identical(openclenv,new("externalptr"))) openclenv = NULL
R/profitLikeModel.R:        magmu=Data$magmu,finesample=finesample, convopt=Data$convopt, openclenv=openclenv, omp_threads=Data$omp_threads,
R/profitLikeModel.R:        magmu=Data$magmu, finesample=finesample, convopt=Data$convopt, openclenv=openclenv, omp_threads=Data$omp_threads,
R/profitUtility.R:profitGetOpenCLEnvs <- function(name="opencl",make.envs=FALSE)
R/profitUtility.R:  openclenvs = data.frame()
R/profitUtility.R:  if(profitHasOpenCL())
R/profitUtility.R:    openclinfo = profitOpenCLEnvInfo()
R/profitUtility.R:    nenvs = length(openclinfo)
R/profitUtility.R:      for(envi in 1:length(openclinfo))
R/profitUtility.R:        openclenv = openclinfo[[envi]]
R/profitUtility.R:        if(length(openclenv$devices)>0){
R/profitUtility.R:          devices = do.call(rbind, lapply(openclenv$devices,
R/profitUtility.R:          devices$env_name = openclenv$name
R/profitUtility.R:          devices$version = openclenv$opencl_version
R/profitUtility.R:          openclenvs = rbind(openclenvs,devices)
R/profitUtility.R:        for(i in 1:nrow(openclenvs)) ptrvec = c(ptrvec, new("externalptr"))
R/profitUtility.R:        openclenvs$env_single = ptrvec
R/profitUtility.R:        openclenvs$env_double = ptrvec
R/profitUtility.R:        for(i in 1:nrow(openclenvs))
R/profitUtility.R:          if (openclenvs$supports_single[i])
R/profitUtility.R:            env = profitOpenCLEnv(openclenvs$env_i[i],openclenvs$dev_i[i],use_double = FALSE)
R/profitUtility.R:              openclenvs$env_single[[i]] = env
R/profitUtility.R:          if (openclenvs$supports_double[i])
R/profitUtility.R:            env = profitOpenCLEnv(openclenvs$env_i[i],openclenvs$dev_i[i],use_double = TRUE)
R/profitUtility.R:              openclenvs$env_double[[i]] = env
R/profitUtility.R:  return(openclenvs)
R/profitUtility.R:  if(profitHasOpenCL()) rv = c(rv,"opencl")
R/profitSetupData.R:  finesample=1L, psf=NULL, fitpsf=FALSE, omp_threads=NULL, openclenv=NULL,
R/profitSetupData.R:  openclenv_int=openclenv, openclenv_conv=openclenv,
R/profitSetupData.R:  benchopenclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
R/profitSetupData.R:      returnfine = TRUE, returncrop = FALSE, openclenv=openclenv_int, omp_threads=omp_threads)
R/profitSetupData.R:      openclenvs = benchopenclenvs, omp_threads = omp_threads, finesample = finesample)
R/profitSetupData.R:    openclenv_int = bestint$openclenv
R/profitSetupData.R:    # Note if openclenv is "get", and openclenv_int isn't specified, it will inherit the "get" value
R/profitSetupData.R:    if(identical(openclenv_int,"get")) openclenv_int = openclenv
R/profitSetupData.R:    # If that's false, it will simply use the passed-in openclenv
R/profitSetupData.R:  convopt = list(convolver=NULL,openclenv=openclenv_conv)
R/profitSetupData.R:        openclenvs = benchopenclenvs, omp_threads = omp_threads)
R/profitSetupData.R:      convopt$openclenv = bestconv$openclenv
R/profitSetupData.R:      if(identical(openclenv_conv,"get")) openclenv_conv = profitOpenCLEnv()
R/profitSetupData.R:        if(is.null(openclenv_conv)) convmethod = "brute"
R/profitSetupData.R:        else convmethod = "opencl"
R/profitSetupData.R:        openclenv=openclenv_conv)
R/profitSetupData.R:    openclenv = openclenv_int)
R/profitSetupData.R:  openclenv=NULL, openclenv_int=openclenv, openclenv_conv=openclenv,
R/profitSetupData.R:  benchopenclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
R/profitSetupData.R:  if (!is.null(openclenv)) {
R/profitSetupData.R:    if (class(openclenv) == "externalptr") {
R/profitSetupData.R:      openclenv = openclenv
R/profitSetupData.R:    else if (identical(openclenv,"get")) {
R/profitSetupData.R:      openclenv = profitOpenCLEnv()
R/profitSetupData.R:    openclenv = openclenv, openclenv_int = openclenv_int, openclenv_conv = openclenv_conv,
R/profitSetupData.R:    benchintprecisions = benchintprecisions, benchopenclenvs = benchopenclenvs,
R/profitSetupData.R:    openclenv=openclenv, 
R/profitConvolver.R:		reuse_psf_fft = TRUE, fft_effort = 0, omp_threads = NULL, openclenv = NULL)
R/profitConvolver.R:	      l(reuse_psf_fft), i(fft_effort), i(omp_threads), openclenv)
R/profitHPC.R:profitOpenCLEnvInfo = function() {
R/profitHPC.R:	.Call('R_profit_openclenv_info')
R/profitHPC.R:profitOpenCLEnv = function(plat_idx=1, dev_idx=1, use_double=FALSE) {
R/profitHPC.R:  tempenvlist=profitOpenCLEnvInfo()
R/profitHPC.R:	.Call('R_profit_openclenv', as.integer(plat_idx-1), as.integer(dev_idx-1), as.integer(use_double))
R/profitHPC.R:profitHasOpenCL = function() {
R/profitHPC.R:  return(!is.null(profitOpenCLEnvInfo()))
vignettes/ProFit-OpenCL-OpenMP.Rmd:title: "ProFit: OpenCL and OpenMP Support"
vignettes/ProFit-OpenCL-OpenMP.Rmd:  %\VignetteIndexEntry{ProFit: OpenCL and OpenMP Support}
vignettes/ProFit-OpenCL-OpenMP.Rmd:# OpenCL support
vignettes/ProFit-OpenCL-OpenMP.Rmd:The support for OpenCL compatible graphics cards is described in a bit of detail in the help for **profitOpenCLEnv** **profitOpenCLEnvInfo**. This should generally work more-or-less out the box if you have a compatible card on your local machine (mileage may vary though). An example using the defaults (that uses the first vaialbel card on your machine if possible):
vignettes/ProFit-OpenCL-OpenMP.Rmd:tempCL=profitOpenCLEnv()
vignettes/ProFit-OpenCL-OpenMP.Rmd:magimage(profitMakeModel(modellist=modellist, dim=c(200,200), openclenv=tempCL))
vignettes/ProFit-OpenCL-OpenMP.Rmd:system.time(profitMakeModel(modellist=modellist, dim=c(2000,2000), openclenv=tempCL))
vignettes/ProFit-OpenCL-OpenMP.Rmd:system.time(profitMakeModel(modellist=modellist, dim=c(2000,2000), openclenv={}))
vignettes/ProFit-OpenCL-OpenMP.Rmd:system.time(for(i in 1:100){profitMakeModel(modellist=modellist, dim=c(200,200), openclenv=tempCL)})
vignettes/ProFit-OpenCL-OpenMP.Rmd:system.time(for(i in 1:100){profitMakeModel(modellist=modellist, dim=c(200,200), openclenv={})})
vignettes/ProFit-OpenCL-OpenMP.Rmd:On my (ASGR's) MacBook Pro circa 2012 with a quad 2.6 GHz Intel Core i7 CPU and a NVIDIA GeForce GT 650M 1024 MB GPU I see a speed up of a factor ~3.5 for the first example (a single big image) and ~4 for the second example (looped smaller images).
vignettes/ProFit-OpenCL-OpenMP.Rmd:You should see some helpful outputs when this starts which indicates whether you are building with OpenCL and OpenMP support, e.g.:
vignettes/ProFit-OpenCL-OpenMP.Rmd:- Found OpenCL headers
vignettes/ProFit-OpenCL-OpenMP.Rmd:- Found OpenCL libs
vignettes/ProFit-OpenCL-OpenMP.Rmd:- Looking for OpenCL version 2.0
vignettes/ProFit-OpenCL-OpenMP.Rmd:- Looking for OpenCL version 1.2
vignettes/ProFit-OpenCL-OpenMP.Rmd:- Compiling with OpenCL 1.2 support
vignettes/ProFit-OpenCL-OpenMP.Rmd:Depending on your use case any of the three strategies might be most sensible. For fitting a single object you will get the most speed-up from using OpenCL or OpenMP. For fitting a large number of galaxies running an embarrassingly parallel **foreach** loop should offer a similar speed-up to OpenMP using the same number of cores, but it will use much more memory (**foreach** effectively copies the session for each core, which produces additional overheads).
README.md:If you do have the build tools, a development version of R, useful permissions, and a bit of bravery then you will be able to install the latest variants directly from the main ICRAR GitHub branch. You need a version of GCC or Clang that supports vaguely modern C++11 features. For Linux users this should be the case for any OS installed in the last 5 years, but for OSX the tool chain tends to be a fair amount older. If you have 10.9 (Mavericks) or newer and the associated X-Code (6+) then you should probably be fine. Other options might be to install a set of more recent tools from Homebrew (see instructions on installing a more modern Clang in the OpenCLMaybe 0.0 and OpenMP document).
configure:#include <OpenCL/opencl.h>
configure:#include <CL/opencl.h>
configure:		echo "-framework OpenCL"
configure:		echo "-lOpenCL"
configure:#include <OpenCL/opencl.h>
configure:#include <CL/opencl.h>
configure:		echo "- Looking for OpenCL version $maj.$min"
configure:#include <OpenCL/opencl.h>
configure:#include <CL/opencl.h>
configure:			echo "- OpenCL version $maj.$min found in header, checking for function $funcname in library"
configure:			echo " The OpenCL headers found by the compiler declare version $maj.$min, but the OpenCL"
configure:			echo " $funcname function, introduced in OpenCL $maj.$min)."
configure:			echo " This possibly indicates that there is a mixed OpenCL installation in your"
configure:			echo " OpenCL installations."
configure:			echo " We will continue looking for lower OpenCL versions, until we find one that"
configure: * C++-compatible OpenCL kernel source code from ${name}
configure:PROFIT_OPENCL=no
configure:PROFIT_OPENCL_MAJOR=
configure:PROFIT_OPENCL_MINOR=
configure:PROFIT_OPENCL_TARGET_VERSION=
configure:	sexpr="$sexpr; `cmakedefine_sed_replacement PROFIT_OPENCL`"
configure:	sexpr="$sexpr; `at_sed_replacement PROFIT_OPENCL_MAJOR`"
configure:	sexpr="$sexpr; `at_sed_replacement PROFIT_OPENCL_MINOR`"
configure:	sexpr="$sexpr; `at_sed_replacement PROFIT_OPENCL_TARGET_VERSION`"
configure:# Check for OpenCL headers, libs and max OpenCL version
configure:if [ -n "${PROFIT_NO_OPENCL}" ]; then
configure:	echo "- Skipping OpenCL detection"
configure:		echo "- Found OpenCL headers"
configure:			echo "- Found OpenCL libs"
configure:		echo "- Compiling with OpenCL $cl_ver_maj.$cl_ver_min support"
configure:		PROFIT_OPENCL=yes
configure:		PROFIT_OPENCL_MAJOR=${cl_ver_maj}
configure:		PROFIT_OPENCL_MINOR=${cl_ver_min}
configure:		PROFIT_OPENCL_TARGET_VERSION=110
configure:		# In OSX 10.8 (Darwin 12.0) the Apple OpenCL platform was already 1.2,
configure:			PROFIT_OPENCL_TARGET_VERSION=120
configure:		echo "- Compiling without OpenCL support"
NEWS:FEATURE: Added OpenCL support
NEWS:FEATURE: GPU support via OpenCL (see vignettes).
man/profitBenchmarkResultBest.Rd:\item{name}{The name of the best method and/or OpenCL environment.}
man/profitBenchmarkResultBest.Rd:\item{openclenv}{Pointer to the best OpenCL environment; see \code{\link{profitOpenCLEnv}}.}
man/profitDataSetOptionsFromBenchmark.Rd:openclenvs = data.frame()
man/profitDataSetOptionsFromBenchmark.Rd:  benchintmethods = "brute", benchopenclenvs = openclenvs,
man/profitDataSetOptionsFromBenchmark.Rd:  benchintmethods = profitAvailableIntegrators(), benchopenclenvs = openclenvs,
man/profitHasOpenCL.Rd:\alias{profitHasOpenCL}
man/profitHasOpenCL.Rd:Check for presence of OpenMP, OpenCL and FFTW
man/profitHasOpenCL.Rd:Simple utilities that check whether package has compile-time OpenMP, OpenCL
man/profitHasOpenCL.Rd:profitHasOpenCL()
man/profitHasOpenCL.Rd:Logical; states whether package has been installed with OpenMP, OpenCL or FFTW
man/profitHasOpenCL.Rd:\code{\link{profitOpenCLEnv}}, \code{\link{profitOpenCLEnvInfo}}
man/profitHasOpenCL.Rd:profitHasOpenCL()
man/profitHasOpenCL.Rd:\keyword{ OpenCL }% __ONLY ONE__ keyword per line
man/profitClearCache.Rd:In particular, FFTW wisdom and OpenCL compiled kernels are cached by libprofit.
man/profitGetOpenCLEnvs.Rd:\name{profitGetOpenCLEnvs}
man/profitGetOpenCLEnvs.Rd:\alias{profitGetOpenCLEnvs}
man/profitGetOpenCLEnvs.Rd:Get available OpenCL environments
man/profitGetOpenCLEnvs.Rd:This function returns a data.frame with information on available OpenCL environments, which can be used to integrate profiles and/or convolve images with CPUs and GPUs and passed on to \code{\link{profitBenchmark}}.
man/profitGetOpenCLEnvs.Rd:profitGetOpenCLEnvs(name = "opencl", make.envs = FALSE)
man/profitGetOpenCLEnvs.Rd:Note, if the sub-list returned by \code{\link{profitOpenCLEnvInfo}} has NULL devices then that openCL device will be skipped when compiling this data.frame.
man/profitGetOpenCLEnvs.Rd:\code{\link{profitBenchmark}}, \code{\link{profitMakeConvolver}}, \code{\link{profitOpenCLEnv}}
man/profitGetOpenCLEnvs.Rd:envs = profitGetOpenCLEnvs(make.envs=FALSE)
man/profitGetOpenCLEnvs.Rd:\concept{ GPU }
man/profitGetOpenCLEnvs.Rd:\concept{ OpenCL }
man/profitOpenCLEnv.Rd:\name{profitOpenCLEnv}
man/profitOpenCLEnv.Rd:\alias{profitOpenCLEnv}
man/profitOpenCLEnv.Rd:Create OpenCL Pointer Object
man/profitOpenCLEnv.Rd:This function returns a legal external pointer to a GPU card that will then be used to compute models.
man/profitOpenCLEnv.Rd:profitOpenCLEnv(plat_idx = 1, dev_idx = 1, use_double = FALSE)
man/profitOpenCLEnv.Rd:The platform index to use for the GPU computation. If in doubt leave as the default (1).
man/profitOpenCLEnv.Rd:The device index within the platform for the GPU computation. If in doubt leave as the default (1).
man/profitOpenCLEnv.Rd:Some computers might have multiple platforms and devices available for GPU computation. The indices used refer to device number N on platform number M. If you have multiple cards then you might have more than one card device on a single platform, or single devices across multiple platforms.
man/profitOpenCLEnv.Rd:If your computer has a single card (or you do not know what platforms and devices means with regards to GPUs) you probably want to leave the values as their defaults.
man/profitOpenCLEnv.Rd:If there is any error building the OpenCL environment object an error is printed and NULL is returned.
man/profitOpenCLEnv.Rd:\code{\link{profitOpenCLEnvInfo}}, \code{\link{profitClearCache}}, \code{\link{profitMakeModel}}, \code{\link{profitSetupData}}
man/profitOpenCLEnv.Rd:tempCL=profitOpenCLEnv()
man/profitOpenCLEnv.Rd:magimage(profitMakeModel(modellist=modellist, dim=c(200,200), openclenv=tempCL))
man/profitOpenCLEnv.Rd:\concept{ GPU }
man/profitOpenCLEnv.Rd:\concept{ OpenCL }
man/profitOpenCLEnvInfo.Rd:\name{profitOpenCLEnvInfo}
man/profitOpenCLEnvInfo.Rd:\alias{profitOpenCLEnvInfo}
man/profitOpenCLEnvInfo.Rd:Discover System Available OpenCL GPUs
man/profitOpenCLEnvInfo.Rd:This helper function discovers all accessible GPUs that can be used by OpenCL.
man/profitOpenCLEnvInfo.Rd:profitOpenCLEnvInfo()
man/profitOpenCLEnvInfo.Rd:The output from this function has to be interpreted by the user to decide which device and platform should be used. There might be one available GPU that is much faster than the others, so some experimentation may be necessary.
man/profitOpenCLEnvInfo.Rd:List; complex structure containing one or more platforms at the highest level, and within each platform a list of one or more devices. Each platform has "name" and "opencl_version" elements, and each device has "name" and "supports_double" elements.
man/profitOpenCLEnvInfo.Rd:  opencl_version = 1.2 (Numeric; OpenCL version)\cr
man/profitOpenCLEnvInfo.Rd:\code{\link{profitOpenCLEnv}}, \code{\link{profitClearCache}} \code{\link{profitMakeModel}}, \code{\link{profitSetupData}}
man/profitOpenCLEnvInfo.Rd:profitOpenCLEnvInfo()
man/profitOpenCLEnvInfo.Rd:\concept{ GPU }% use one of  RShowDoc("KEYWORDS")
man/profitOpenCLEnvInfo.Rd:\concept{ OpenCL }% __ONLY ONE__ keyword per line
man/profitSetupData.Rd:like.func = "norm", magmu = FALSE, verbose = FALSE, omp_threads = NULL, openclenv = NULL,
man/profitSetupData.Rd:openclenv_int = openclenv, openclenv_conv = openclenv, nbenchmark = 0L,
man/profitSetupData.Rd:benchopenclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
man/profitSetupData.Rd:  An integer indicating the number of threads to use to evaluate radial profiles. If not given only one thread is used. \option{openclenv} has precedence over this option, so if both are given then OpenCL evaluation takes place.
man/profitSetupData.Rd:  \item{openclenv}{
man/profitSetupData.Rd:  If NULL (default) then the CPU is used to compute the profile. If \option{openclenv} is a legal pointer to a graphics card of class externalptr then that card will be used to make a GPU based model. This object can be obtained from the \code{\link{profitOpenCLEnv}} function directly. If \option{openclenv}='get' then the OpenCL environment is obtained from running \code{\link{profitOpenCLEnv}} with default values (which are usually reasonable).
man/profitSetupData.Rd:  \item{openclenv_int}{
man/profitSetupData.Rd:    The OpenCL environment to use for integrating profiles. Defaults to the value specified in \option{openclenv}.
man/profitSetupData.Rd:  \item{openclenv_conv}{
man/profitSetupData.Rd:    The OpenCL environment to use for PSF convolution. Defaults to the value specified in \option{openclenv}.
man/profitSetupData.Rd:  \item{benchopenclenvs}{
man/profitSetupData.Rd:  List of OpenCL environments to benchmark. Defaults to all available environments. The optimal environment will then be used for \option{openclenvint} and \option{openclenvconv}, overriding any values set there.
man/profitSetupData.Rd:\item{convopt}{List including the optimal convolver object and its OpenCL environment (if any).}
man/profitSetupData.Rd:# with OpenCL and OpenMP:
man/profitSetupData.Rd:openclenvs = profitGetOpenCLEnvs(make.envs = TRUE)
man/profitSetupData.Rd:benchintmethods = profitAvailableIntegrators(), benchopenclenvs = openclenvs,
man/profitMakeConvolver.Rd: reuse_psf_fft = TRUE, fft_effort = 0, omp_threads = NULL, openclenv = NULL)
man/profitMakeConvolver.Rd:  \item{openclenv}{
man/profitMakeConvolver.Rd:A valid pointer to an OpenCL environment (obtained from the \code{\link{profitOpenCLEnv}}).
man/profitMakeConvolver.Rd:Used only if \option{type} is \code{"opencl"} or \code{"opencl-local"}
man/profitMakeConvolver.Rd:and ProFit has OpenCL support.
man/profitMakeConvolver.Rd:\code{\link{profitOpenCLEnv}}
man/profitMakeConvolver.Rd:has_openCL=profitHasOpenCL()
man/profitMakeConvolver.Rd:if(has_openCL){
man/profitMakeConvolver.Rd:  convolver_bruteCL = profitMakeConvolver("opencl", c(400, 400), psf,
man/profitMakeConvolver.Rd:  openclenv=profitOpenCLEnv())
man/profitMakeConvolver.Rd:if(has_openCL){
man/profitMakeConvolver.Rd:if(has_openCL){
man/profitBenchmark.Rd:  openclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
man/profitBenchmark.Rd:  \item{openclenvs}{
man/profitBenchmark.Rd:  A data.frame with information on available OpenCL environments, such as that returned from \code{\link{profitGetOpenCLEnvs}}.
man/profitBenchmark.Rd:\item{result}{The benchmarking results in a data.frame; see \code{\link{profitGetOpenCLEnvs}} for more information on the format.}
man/profitBenchmark.Rd:# Use OpenCL if available
man/profitBenchmark.Rd:# Makes a list of available OpenCL environments optionally with double precision if all
man/profitBenchmark.Rd:openclenvs = profitGetOpenCLEnvs(make.envs=TRUE)
man/profitBenchmark.Rd:  bench=profitBenchmark(model.image, modellist=model, nbench=nbench, openclenvs=openclenvs,
man/profitBenchmark.Rd:  bench=profitBenchmark(model.image, psf=psf, nbench=nbench, openclenvs=openclenvs,
man/profitBenchmarkResultStripPointers.Rd: \code{\link{profitBenchmark}}, \code{\link{profitGetOpenCLEnvs}}
man/profitBenchmarkResultStripPointers.Rd:  openclenvs = profitGetOpenCLEnvs(make.envs=TRUE)
man/profitBenchmarkResultStripPointers.Rd:  print(profitBenchmarkResultStripPointers(openclenvs))
man/profitDataBenchmark.Rd:  finesample=1L, psf=NULL, fitpsf=FALSE, omp_threads=NULL, openclenv=NULL,
man/profitDataBenchmark.Rd:  openclenv_int=openclenv, openclenv_conv=openclenv,
man/profitDataBenchmark.Rd:  benchopenclenvs = profitGetOpenCLEnvs(make.envs = TRUE),
man/profitDataBenchmark.Rd:  An integer indicating the number of threads to use to evaluate radial profiles. If not given only one thread is used. \option{openclenv} has precedence over this option, so if both are given then OpenCL evaluation takes place.
man/profitDataBenchmark.Rd:  \item{openclenv}{
man/profitDataBenchmark.Rd:  If NULL (default) then the CPU is used to compute the profile. If \option{openclenv} is a legal pointer to a graphics card of class externalptr then that card will be used to make a GPU based model. This object can be obtained from the \code{\link{profitGetOpenCLEnvs}} function with the make.envs option set to TRUE. If \option{openclenv}='get' then the OpenCL environment is obtained from running \code{\link{profitOpenCLEnv}} with default values (which are usually reasonable).
man/profitDataBenchmark.Rd:  \item{openclenv_int}{
man/profitDataBenchmark.Rd:    The OpenCL environment to use for integrating profiles. Defaults to the value specified in \option{openclenv}.
man/profitDataBenchmark.Rd:  \item{openclenv_conv}{
man/profitDataBenchmark.Rd:    The OpenCL environment to use for PSF convolution. Defaults to the value specified in \option{openclenv}.
man/profitDataBenchmark.Rd:  \item{benchopenclenvs}{
man/profitDataBenchmark.Rd:  List of OpenCL environments to benchmark. Defaults to all available environments. The optimal environment will then be used for \option{openclenvint} and \option{openclenvconv}, overriding any values set there.
man/profitDataBenchmark.Rd:openclenvs = data.frame()
man/profitDataBenchmark.Rd:  benchintmethods = "brute", benchopenclenvs = openclenvs,
man/profitDataBenchmark.Rd:  benchintmethods = profitAvailableIntegrators(), benchopenclenvs = openclenvs,
man/profitAvailableIntegrators.Rd:"opencl", for consistency with \code{\link{profitAvailableConvolvers}}.
man/profitAvailableIntegrators.Rd:but indirectly via OpenCL environment variables.
man/profitAvailableIntegrators.Rd:\code{\link{profitHasOpenCL}},
man/profitMakeModel.Rd:  rescaleflux = FALSE, convopt = NULL, psfdim = c(25, 25), openclenv = NULL,
man/profitMakeModel.Rd:  \item{openclenv}{
man/profitMakeModel.Rd:  If NULL (default) then the CPU is used to compute the profile. If \option{openclenv} is a legal pointer to a graphics card of class externalptr then that card will be used to make a GPU based model. This object can be obtained from the \code{\link{profitOpenCLEnv}} function directly. If \option{openclenv}='get' then the OpenCL environment is obtained from running \code{\link{profitOpenCLEnv}} with default values (which are usually reasonable).
man/profitMakeModel.Rd:  An integer indicating the number of threads to use to evaluate radial profiles. If not given only one thread is used. \option{openclenv} has precedence over this option, so if both are given then OpenCL evaluation takes place.
man/profitMakeModel.Rd:# Using a GPU to create the image:
man/profitMakeModel.Rd:tempCL=profitOpenCLEnv()
man/profitMakeModel.Rd:profitMakeModel(modellist=modellist, dim=c(200,200), openclenv=tempCL, plot=TRUE)
man/profitMakeModel.Rd:# OpenCL due to the large number of system calls made to the GPU.
man/profitMakeModel.Rd:openclenv=tempCL)})
man/profitAvailableConvolvers.Rd:\code{\link{profitHasOpenCL}},
src/libprofit/src/profit/brokenexponential.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/brokenexponential.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/opencl_impl.h: * Internal header file for OpenCL functionality
src/libprofit/src/profit/opencl_impl.h:#ifndef PROFIT_OPENCL_IMPL_H
src/libprofit/src/profit/opencl_impl.h:#define PROFIT_OPENCL_IMPL_H
src/libprofit/src/profit/opencl_impl.h:#include "profit/opencl.h"
src/libprofit/src/profit/opencl_impl.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/opencl_impl.h:/* Quickly fail for OpenCL < 1.1 */
src/libprofit/src/profit/opencl_impl.h:# if !defined(PROFIT_OPENCL_MAJOR) || !defined(PROFIT_OPENCL_MINOR)
src/libprofit/src/profit/opencl_impl.h:#  error "No OpenCL version specified"
src/libprofit/src/profit/opencl_impl.h:# elif PROFIT_OPENCL_MAJOR < 1 || (PROFIT_OPENCL_MAJOR == 1 && PROFIT_OPENCL_MINOR < 1 )
src/libprofit/src/profit/opencl_impl.h:#  error "libprofit requires at minimum OpenCL >= 1.1"
src/libprofit/src/profit/opencl_impl.h:/* MacOS 10.14 (Mojave) started deprecating OpenCL entirely */
src/libprofit/src/profit/opencl_impl.h:/* Define the target OpenCL version based on the given major/minor version */
src/libprofit/src/profit/opencl_impl.h:# define CL_HPP_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
src/libprofit/src/profit/opencl_impl.h:# define CL_TARGET_OPENCL_VERSION  MAKE_VERSION(PROFIT_OPENCL_MAJOR, PROFIT_OPENCL_MINOR)
src/libprofit/src/profit/opencl_impl.h:# define CL_HPP_MINIMUM_OPENCL_VERSION PROFIT_OPENCL_TARGET_VERSION
src/libprofit/src/profit/opencl_impl.h: * given event as an OpenCL_comand_times structure.
src/libprofit/src/profit/opencl_impl.h:OpenCL_command_times cl_cmd_times(const cl::Event &evt);
src/libprofit/src/profit/opencl_impl.h:class OpenCLEnvImpl;
src/libprofit/src/profit/opencl_impl.h:typedef std::shared_ptr<OpenCLEnvImpl> OpenCLEnvImplPtr;
src/libprofit/src/profit/opencl_impl.h: * An OpenCL environment
src/libprofit/src/profit/opencl_impl.h:class OpenCLEnvImpl : public OpenCLEnv {
src/libprofit/src/profit/opencl_impl.h:	OpenCLEnvImpl(cl::Device device, cl_ver_t version, cl::Context context,
src/libprofit/src/profit/opencl_impl.h:	static OpenCLEnvImplPtr fromOpenCLEnvPtr(const OpenCLEnvPtr &ptr) {
src/libprofit/src/profit/opencl_impl.h:		return std::static_pointer_cast<OpenCLEnvImpl>(ptr);
src/libprofit/src/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
src/libprofit/src/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
src/libprofit/src/profit/opencl_impl.h:	// Implementing OpenCL_env's interface
src/libprofit/src/profit/opencl_impl.h:	 * Returns the amount of memory, in bytes, that each OpenCL Compute Unit
src/libprofit/src/profit/opencl_impl.h:	 * by this OpenCL environment.
src/libprofit/src/profit/opencl_impl.h:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/opencl_impl.h:	/** The device to be used throughout OpenCL operations */
src/libprofit/src/profit/opencl_impl.h:	/** The OpenCL supported by the platform this device belongs to */
src/libprofit/src/profit/opencl_impl.h:	/** The OpenCL context used throughout the OpenCL operations */
src/libprofit/src/profit/opencl_impl.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/opencl_impl.h:#endif /* PROFIT_OPENCL_IMPL_H */
src/libprofit/src/profit/opencl.h: * User-facing header file for OpenCL functionality
src/libprofit/src/profit/opencl.h:#ifndef PROFIT_OPENCL_H
src/libprofit/src/profit/opencl.h:#define PROFIT_OPENCL_H
src/libprofit/src/profit/opencl.h: * A datatype for storing an OpenCL version.
src/libprofit/src/profit/opencl.h: * It should have the form major*100 + minor*10 (e.g., 120 for OpenCL 1.2)
src/libprofit/src/profit/opencl.h: * A structure holding two times associated with OpenCL commands:
src/libprofit/src/profit/opencl.h:class PROFIT_API OpenCL_command_times {
src/libprofit/src/profit/opencl.h:	OpenCL_command_times &operator+=(const OpenCL_command_times &other);
src/libprofit/src/profit/opencl.h:	const OpenCL_command_times operator+(const OpenCL_command_times &other) const;
src/libprofit/src/profit/opencl.h: * A structure holding a number of OpenCL command times (filling, writing,
src/libprofit/src/profit/opencl.h: * kernel and reading) plus other OpenCL-related times.
src/libprofit/src/profit/opencl.h:struct PROFIT_API OpenCL_times {
src/libprofit/src/profit/opencl.h:	OpenCL_command_times writing_times;
src/libprofit/src/profit/opencl.h:	OpenCL_command_times reading_times;
src/libprofit/src/profit/opencl.h:	OpenCL_command_times filling_times;
src/libprofit/src/profit/opencl.h:	OpenCL_command_times kernel_times;
src/libprofit/src/profit/opencl.h: * An OpenCL environment.
src/libprofit/src/profit/opencl.h:class PROFIT_API OpenCLEnv {
src/libprofit/src/profit/opencl.h:	virtual ~OpenCLEnv() {};
src/libprofit/src/profit/opencl.h:	 * Returns the maximum OpenCL version supported by the underlying device.
src/libprofit/src/profit/opencl.h:	 * Returns the name of the OpenCL platform of this environment.
src/libprofit/src/profit/opencl.h:	 * @return The name of the OpenCL platform
src/libprofit/src/profit/opencl.h:	 * Returns the name of the OpenCL device of this environment.
src/libprofit/src/profit/opencl.h:	 * @return The name of the OpenCL device
src/libprofit/src/profit/opencl.h:/// Handy typedef for shared pointers to OpenCL_env objects
src/libprofit/src/profit/opencl.h:typedef std::shared_ptr<OpenCLEnv> OpenCLEnvPtr;
src/libprofit/src/profit/opencl.h: * A structure holding information about a specific OpenCL device
src/libprofit/src/profit/opencl.h:typedef struct PROFIT_API _OpenCL_dev_info {
src/libprofit/src/profit/opencl.h:	/** The OpenCL version supported by this device */
src/libprofit/src/profit/opencl.h:} OpenCL_dev_info;
src/libprofit/src/profit/opencl.h: * An structure holding information about a specific OpenCL platform.
src/libprofit/src/profit/opencl.h:typedef struct PROFIT_API _OpenCL_plat_info {
src/libprofit/src/profit/opencl.h:	/** The supported OpenCL version */
src/libprofit/src/profit/opencl.h:	cl_ver_t supported_opencl_version;
src/libprofit/src/profit/opencl.h:	std::map<int, OpenCL_dev_info> dev_info;
src/libprofit/src/profit/opencl.h:} OpenCL_plat_info;
src/libprofit/src/profit/opencl.h: * Queries the system about the OpenCL supported platforms and devices and returns
src/libprofit/src/profit/opencl.h: *         OpenCL platforms found on this system.
src/libprofit/src/profit/opencl.h:PROFIT_API std::map<int, OpenCL_plat_info> get_opencl_info();
src/libprofit/src/profit/opencl.h: * Prepares an OpenCL working space for using with libprofit.
src/libprofit/src/profit/opencl.h: * the libprofit OpenCL kernel sources to be used against it, and set up a queue
src/libprofit/src/profit/opencl.h: * @param enable_profiling Whether OpenCL profiling capabilities should be
src/libprofit/src/profit/opencl.h: *        turned on in the OpenCL Queue created within this envinronment.
src/libprofit/src/profit/opencl.h: * @return A pointer to a OpenCL_env structure, which contains the whole set of
src/libprofit/src/profit/opencl.h:PROFIT_API OpenCLEnvPtr get_opencl_environment(
src/libprofit/src/profit/opencl.h:#endif /* PROFIT_OPENCL_H */
src/libprofit/src/profit/exceptions.cpp:opencl_error::opencl_error(const std::string &what_arg) :
src/libprofit/src/profit/exceptions.cpp:opencl_error::~opencl_error() throw () {
src/libprofit/src/profit/ferrer.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/ferrer.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/king.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/king.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/radial.cpp:#include "profit/opencl.h"
src/libprofit/src/profit/radial.cpp:#ifndef PROFIT_OPENCL
src/libprofit/src/profit/radial.cpp:	 * We fallback to the CPU implementation if no OpenCL context has been
src/libprofit/src/profit/radial.cpp:	 * given, or if there is no OpenCL kernel implementing the profile
src/libprofit/src/profit/radial.cpp:	auto env = OpenCLEnvImpl::fromOpenCLEnvPtr(model.get_opencl_env());
src/libprofit/src/profit/radial.cpp:	if( force_cpu || !env || !supports_opencl() ) {
src/libprofit/src/profit/radial.cpp:			evaluate_opencl<double>(image, mask, scale, env);
src/libprofit/src/profit/radial.cpp:			evaluate_opencl<float>(image, mask, scale, env);
src/libprofit/src/profit/radial.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/libprofit/src/profit/radial.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/radial.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/radial.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/radial.cpp:void RadialProfile::evaluate_opencl(Image &image, const Mask & /*mask*/, const PixelScale &scale, OpenCLEnvImplPtr &env) {
src/libprofit/src/profit/radial.cpp:	OpenCL_times cl_times0 {};
src/libprofit/src/profit/radial.cpp:	OpenCL_times ss_cl_times {};
src/libprofit/src/profit/radial.cpp:	system_clock::time_point t0, t_kprep, t_opencl, t_loopstart, t_loopend, t_imgtrans;
src/libprofit/src/profit/radial.cpp:	// OpenCL 1.2 allows to do this; otherwise the work has to be done in the kernel
src/libprofit/src/profit/radial.cpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/radial.cpp:#endif /* CL_HPP_TARGET_OPENCL_VERSION >= 120 */
src/libprofit/src/profit/radial.cpp:	t_opencl = system_clock::now();
src/libprofit/src/profit/radial.cpp:	stats->final_image += to_nsecs(system_clock::now() - t_opencl);
src/libprofit/src/profit/radial.cpp:	/* These are the OpenCL-related timings so far */
src/libprofit/src/profit/radial.cpp:	cl_times0.total = to_nsecs(t_opencl - t_kprep);
src/libprofit/src/profit/radial.cpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/radial.cpp:#endif /* CL_HPP_TARGET_OPENCL_VERSION >= 120 */
src/libprofit/src/profit/radial.cpp:		system_clock::time_point t0, t_newsamples, t_trans_h2k, t_kprep, t_opencl, t_trans_k2h;
src/libprofit/src/profit/radial.cpp:			t_opencl = system_clock::now();
src/libprofit/src/profit/radial.cpp:			stats->subsampling.final_transform += to_nsecs(t_trans_k2h - t_opencl);
src/libprofit/src/profit/radial.cpp:			ss_cl_times.total += to_nsecs(t_opencl - t_trans_h2k);
src/libprofit/src/profit/radial.cpp:	stats->subsampling.pre_subsampling = to_nsecs(t_loopstart - t_opencl);
src/libprofit/src/profit/radial.cpp:bool RadialProfile::supports_opencl() const {
src/libprofit/src/profit/radial.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/radial.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/radial.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/exceptions.h: * Exception class thrown when an error occurs while dealing with OpenCL.
src/libprofit/src/profit/exceptions.h:class PROFIT_API opencl_error : public exception
src/libprofit/src/profit/exceptions.h:	explicit opencl_error(const std::string &what);
src/libprofit/src/profit/exceptions.h:	~opencl_error() throw();
src/libprofit/src/profit/config.h.in:/** Whether libprofit contains OpenCL support */
src/libprofit/src/profit/config.h.in:#cmakedefine PROFIT_OPENCL
src/libprofit/src/profit/config.h.in: * If OpenCL support is present, the major OpenCL version supported by
src/libprofit/src/profit/config.h.in:#define PROFIT_OPENCL_MAJOR @PROFIT_OPENCL_MAJOR@
src/libprofit/src/profit/config.h.in: * If OpenCL support is present, the minor OpenCL version supported by
src/libprofit/src/profit/config.h.in:#define PROFIT_OPENCL_MINOR @PROFIT_OPENCL_MINOR@
src/libprofit/src/profit/config.h.in: * If OpenCL support is present, the target OpenCL version supported by
src/libprofit/src/profit/config.h.in:#define PROFIT_OPENCL_TARGET_VERSION @PROFIT_OPENCL_TARGET_VERSION@
src/libprofit/src/profit/convolve.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/convolve.cpp:OpenCLConvolver::OpenCLConvolver(OpenCLEnvImplPtr opencl_env) :
src/libprofit/src/profit/convolve.cpp:	env(std::move(opencl_env))
src/libprofit/src/profit/convolve.cpp:		throw invalid_parameter("Empty OpenCL environment given to OpenCLConvolver");
src/libprofit/src/profit/convolve.cpp:Image OpenCLConvolver::convolve_impl(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out)
src/libprofit/src/profit/convolve.cpp:		os << "OpenCL error while convolving: " << e.what() << ". OpenCL error code: " << e.err();
src/libprofit/src/profit/convolve.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/convolve.cpp:Dimensions OpenCLConvolver::cl_padding(const Dimensions &src_dims) const
src/libprofit/src/profit/convolve.cpp:PointPair OpenCLConvolver::padding(const Dimensions &src_dims, const Dimensions &/*krn_dims*/) const
src/libprofit/src/profit/convolve.cpp:Image OpenCLConvolver::_convolve(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out) {
src/libprofit/src/profit/convolve.cpp:Image OpenCLConvolver::_clpadded_convolve(const Image &src, const Image &krn, const Image &orig_src) {
src/libprofit/src/profit/convolve.cpp:OpenCLLocalConvolver::OpenCLLocalConvolver(OpenCLEnvImplPtr opencl_env) :
src/libprofit/src/profit/convolve.cpp:	env(std::move(opencl_env))
src/libprofit/src/profit/convolve.cpp:		throw invalid_parameter("Empty OpenCL environment given to OpenCLLocalConvolver");
src/libprofit/src/profit/convolve.cpp:Image OpenCLLocalConvolver::convolve_impl(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out)
src/libprofit/src/profit/convolve.cpp:		os << "OpenCL error while convolving: " << e.what() << ". OpenCL error code: " << e.err();
src/libprofit/src/profit/convolve.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/convolve.cpp:Image OpenCLLocalConvolver::_convolve(const Image &src, const Image &krn, const Mask &mask, bool crop, Point &offset_out) {
src/libprofit/src/profit/convolve.cpp:Image OpenCLLocalConvolver::_clpadded_convolve(const Image &src, const Image &krn, const Image &orig_src) {
src/libprofit/src/profit/convolve.cpp:		os << "Not enough local memory available for OpenCL local 2D convolution. ";
src/libprofit/src/profit/convolve.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/convolve.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/convolve.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/convolve.cpp:		case OPENCL:
src/libprofit/src/profit/convolve.cpp:			return std::make_shared<OpenCLConvolver>(OpenCLEnvImpl::fromOpenCLEnvPtr(prefs.opencl_env));
src/libprofit/src/profit/convolve.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/convolve.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/convolve.cpp:	else if (type == "opencl") {
src/libprofit/src/profit/convolve.cpp:		return create_convolver(OPENCL, prefs);
src/libprofit/src/profit/convolve.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/king.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/king.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/model.h:#include "profit/opencl.h"
src/libprofit/src/profit/model.h:	void set_opencl_env(const OpenCLEnvPtr &opencl_env) {
src/libprofit/src/profit/model.h:		this->opencl_env = opencl_env;
src/libprofit/src/profit/model.h:	OpenCLEnvPtr get_opencl_env() const {
src/libprofit/src/profit/model.h:		return opencl_env;
src/libprofit/src/profit/model.h:	OpenCLEnvPtr opencl_env;
src/libprofit/src/profit/radial.h:#include "profit/opencl_impl.h"
src/libprofit/src/profit/radial.h:	/// Whether the CPU evaluation method should be used, even if an OpenCL
src/libprofit/src/profit/radial.h:	/// environment has been given (and libprofit has been compiled with OpenCL support)
src/libprofit/src/profit/radial.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/radial.h:	 * Indicates whether this profile supports OpenCL evaluation or not
src/libprofit/src/profit/radial.h:	 * (i.e., implements the required OpenCL kernels)
src/libprofit/src/profit/radial.h:	 * @return Whether this profile supports OpenCL evaluation. The default
src/libprofit/src/profit/radial.h:	virtual bool supports_opencl() const;
src/libprofit/src/profit/radial.h:	/* Evaluates this radial profile using an OpenCL kernel and floating type FT */
src/libprofit/src/profit/radial.h:	void evaluate_opencl(Image &image, const Mask &mask, const PixelScale &scale, OpenCLEnvImplPtr &env);
src/libprofit/src/profit/radial.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/coresersic.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/coresersic.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/opencl.cpp: * OpenCL utility methods for libprofit
src/libprofit/src/profit/opencl.cpp:#include "profit/opencl_impl.h"
src/libprofit/src/profit/opencl.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/opencl.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/opencl.cpp:OpenCL_command_times &OpenCL_command_times::operator+=(const OpenCL_command_times &other) {
src/libprofit/src/profit/opencl.cpp:const OpenCL_command_times OpenCL_command_times::operator+(const OpenCL_command_times &other) const {
src/libprofit/src/profit/opencl.cpp:	OpenCL_command_times t1 = *this;
src/libprofit/src/profit/opencl.cpp:// Simple implementation of public methods for non-OpenCL builds
src/libprofit/src/profit/opencl.cpp:#ifndef PROFIT_OPENCL
src/libprofit/src/profit/opencl.cpp:std::map<int, OpenCL_plat_info> get_opencl_info() {
src/libprofit/src/profit/opencl.cpp:	return std::map<int, OpenCL_plat_info>();
src/libprofit/src/profit/opencl.cpp:OpenCLEnvPtr get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling)
src/libprofit/src/profit/opencl.cpp:// Functions to read the duration of OpenCL events (queue->submit and start->end)
src/libprofit/src/profit/opencl.cpp:OpenCL_command_times cl_cmd_times(const cl::Event &evt) {
src/libprofit/src/profit/opencl.cpp:static cl_ver_t get_opencl_version(const std::string &version)
src/libprofit/src/profit/opencl.cpp:	// Version string should be of type "OpenCL<space><major_version.minor_version><space><platform-specific information>"
src/libprofit/src/profit/opencl.cpp:	if( version.find("OpenCL ") != 0) {
src/libprofit/src/profit/opencl.cpp:		throw opencl_error(std::string("OpenCL version string doesn't start with 'OpenCL ': ") + version);
src/libprofit/src/profit/opencl.cpp:	auto opencl_version = version.substr(7, next_space);
src/libprofit/src/profit/opencl.cpp:	auto dot_idx = opencl_version.find(".");
src/libprofit/src/profit/opencl.cpp:	if( dot_idx == opencl_version.npos ) {
src/libprofit/src/profit/opencl.cpp:		throw opencl_error("OpenCL version doesn't contain a dot: " + opencl_version);
src/libprofit/src/profit/opencl.cpp:	auto major = stoui(opencl_version.substr(0, dot_idx));
src/libprofit/src/profit/opencl.cpp:	auto minor = stoui(opencl_version.substr(dot_idx+1, opencl_version.npos));
src/libprofit/src/profit/opencl.cpp:static cl_ver_t get_opencl_version(const cl::Platform &platform) {
src/libprofit/src/profit/opencl.cpp:	return get_opencl_version(platform.getInfo<CL_PLATFORM_VERSION>());
src/libprofit/src/profit/opencl.cpp:static cl_ver_t get_opencl_version(const cl::Device &device) {
src/libprofit/src/profit/opencl.cpp:	return get_opencl_version(device.getInfo<CL_DEVICE_VERSION>());
src/libprofit/src/profit/opencl.cpp:	if (get_opencl_version(device) < 120) {
src/libprofit/src/profit/opencl.cpp:std::map<int, OpenCL_plat_info> _get_opencl_info() {
src/libprofit/src/profit/opencl.cpp:	std::map<int, OpenCL_plat_info> pinfo;
src/libprofit/src/profit/opencl.cpp:		std::map<int, OpenCL_dev_info> dinfo;
src/libprofit/src/profit/opencl.cpp:			dinfo[didx++] = OpenCL_dev_info{
src/libprofit/src/profit/opencl.cpp:				get_opencl_version(device),
src/libprofit/src/profit/opencl.cpp:		pinfo[pidx++] = OpenCL_plat_info{name, get_opencl_version(platform), dinfo};
src/libprofit/src/profit/opencl.cpp:std::map<int, OpenCL_plat_info> get_opencl_info() {
src/libprofit/src/profit/opencl.cpp:		return _get_opencl_info();
src/libprofit/src/profit/opencl.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/libprofit/src/profit/opencl.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/opencl.cpp:	auto plat_part = valid_fname(plat.getInfo<CL_PLATFORM_NAME>()) + "_" + std::to_string(get_opencl_version(plat));
src/libprofit/src/profit/opencl.cpp:	auto dev_part = valid_fname(device.getInfo<CL_DEVICE_NAME>()) + "_" + std::to_string(get_opencl_version(device));
src/libprofit/src/profit/opencl.cpp:	auto the_dir = create_dirs(get_profit_home(), {std::string("opencl_cache"), plat_part});
src/libprofit/src/profit/opencl.cpp:			throw opencl_error(os.str());
src/libprofit/src/profit/opencl.cpp:		// Some OpenCL drivers (e.g., MacOS Mojave Intel) don't correctly build from binaries
src/libprofit/src/profit/opencl.cpp:		throw opencl_error("Error while getting OpenCL platforms");
src/libprofit/src/profit/opencl.cpp:		throw opencl_error("No platforms found. Check OpenCL installation");
src/libprofit/src/profit/opencl.cpp:		ss << "OpenCL platform index " << platform_idx << " must be < " << all_platforms.size();
src/libprofit/src/profit/opencl.cpp:		throw opencl_error("No devices found. Check OpenCL installation");
src/libprofit/src/profit/opencl.cpp:		ss << "OpenCL device index " << device_idx << " must be < " << all_devices.size();
src/libprofit/src/profit/opencl.cpp:		throw opencl_error("Double precision requested but not supported by device");
src/libprofit/src/profit/opencl.cpp:OpenCLEnvPtr _get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling) {
src/libprofit/src/profit/opencl.cpp:	return std::make_shared<OpenCLEnvImpl>(device, get_opencl_version(device), context, queue, program, use_double, enable_profiling);
src/libprofit/src/profit/opencl.cpp:OpenCLEnvPtr get_opencl_environment(unsigned int platform_idx, unsigned int device_idx, bool use_double, bool enable_profiling) {
src/libprofit/src/profit/opencl.cpp:		return _get_opencl_environment(platform_idx, device_idx, use_double, enable_profiling);
src/libprofit/src/profit/opencl.cpp:		os << "OpenCL error: " << e.what() << ". OpenCL error code: " << e.err();
src/libprofit/src/profit/opencl.cpp:		throw opencl_error(os.str());
src/libprofit/src/profit/opencl.cpp:unsigned long OpenCLEnvImpl::max_local_memory() {
src/libprofit/src/profit/opencl.cpp:unsigned int OpenCLEnvImpl::compute_units() {
src/libprofit/src/profit/opencl.cpp:cl::Event OpenCLEnvImpl::queue_write(const cl::Buffer &buffer, const void *data, const std::vector<cl::Event>* wait_evts) {
src/libprofit/src/profit/opencl.cpp:cl::Event OpenCLEnvImpl::queue_kernel(const cl::Kernel &kernel, const cl::NDRange global, const std::vector<cl::Event>* wait_evts, const cl::NDRange &local) {
src/libprofit/src/profit/opencl.cpp:cl::Event OpenCLEnvImpl::queue_read(const cl::Buffer &buffer, void *data, const std::vector<cl::Event>* wait_evts) {
src/libprofit/src/profit/opencl.cpp:cl::Kernel OpenCLEnvImpl::get_kernel(const std::string &name) {
src/libprofit/src/profit/opencl.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/profile.h:#include "profit/opencl.h"
src/libprofit/src/profit/profile.h:	OpenCL_times cl_times;
src/libprofit/src/profit/profile.h:	OpenCL_times cl_times;
src/libprofit/src/profit/profit.h:#include "profit/opencl.h"
src/libprofit/src/profit/moffat.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/moffat.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/convolve.h:#include "profit/opencl.h"
src/libprofit/src/profit/convolve.h:	/// @copydoc OpenCLConvolver
src/libprofit/src/profit/convolve.h:	OPENCL,
src/libprofit/src/profit/convolve.h:		opencl_env(),
src/libprofit/src/profit/convolve.h:	    OpenCLEnvPtr opencl_env, effort_t effort, bool reuse_krn_fft,
src/libprofit/src/profit/convolve.h:		opencl_env(opencl_env),
src/libprofit/src/profit/convolve.h:	/// A pointer to an OpenCL environment. Used by the OPENCL convolvers.
src/libprofit/src/profit/convolve.h:	OpenCLEnvPtr opencl_env;
src/libprofit/src/profit/moffat.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/moffat.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/brokenexponential.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/brokenexponential.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/library.cpp:bool has_opencl()
src/libprofit/src/profit/library.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:unsigned short opencl_version_major()
src/libprofit/src/profit/library.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:	return PROFIT_OPENCL_MAJOR;
src/libprofit/src/profit/library.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:unsigned short opencl_version_minor()
src/libprofit/src/profit/library.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:	return PROFIT_OPENCL_MINOR;
src/libprofit/src/profit/library.cpp:#endif // PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/library.cpp:	auto opencl_cache = profit_home + "/opencl_cache";
src/libprofit/src/profit/library.cpp:	if (dir_exists(opencl_cache)) {
src/libprofit/src/profit/library.cpp:		recursive_remove(opencl_cache);
src/libprofit/src/profit/coresersic.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/coresersic.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/ferrer.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/ferrer.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/sersic.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/sersic.h:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/convolver_impl.h:#include "profit/opencl_impl.h"
src/libprofit/src/profit/convolver_impl.h:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/convolver_impl.h: * A brute-force convolver that is implemented using OpenCL
src/libprofit/src/profit/convolver_impl.h: * Depending on the floating-point support found at runtime in the given OpenCL
src/libprofit/src/profit/convolver_impl.h:class OpenCLConvolver : public Convolver {
src/libprofit/src/profit/convolver_impl.h:	explicit OpenCLConvolver(OpenCLEnvImplPtr opencl_env);
src/libprofit/src/profit/convolver_impl.h:	OpenCLEnvImplPtr env;
src/libprofit/src/profit/convolver_impl.h:	// returns the extra OpenCL-imposed padding
src/libprofit/src/profit/convolver_impl.h: * Like OpenCLConvolver, but uses a local memory cache
src/libprofit/src/profit/convolver_impl.h:class OpenCLLocalConvolver : public Convolver {
src/libprofit/src/profit/convolver_impl.h:	explicit OpenCLLocalConvolver(OpenCLEnvImplPtr opencl_env);
src/libprofit/src/profit/convolver_impl.h:	OpenCLEnvImplPtr env;
src/libprofit/src/profit/convolver_impl.h:#endif // PROFIT_OPENCL
src/libprofit/src/profit/cl/brokenexponential-double.cl: * Double-precision Broken Exponential profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/brokenexponential-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/convolve-float.cl: * float 2D convolution OpenCL implementation for libprofit
src/libprofit/src/profit/cl/ferrer-float.cl: * Single-precision Ferrer profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/ferrer-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/cl2.hpp: *   \brief C++ bindings for OpenCL 1.0 (rev 48), OpenCL 1.1 (rev 33),
src/libprofit/src/profit/cl/cl2.hpp: *       OpenCL 1.2 (rev 15) and OpenCL 2.0 (rev 29)
src/libprofit/src/profit/cl/cl2.hpp: *   Derived from the OpenCL 1.x C++ bindings written by
src/libprofit/src/profit/cl/cl2.hpp: *       http://khronosgroup.github.io/OpenCL-CLHPP/
src/libprofit/src/profit/cl/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP/releases
src/libprofit/src/profit/cl/cl2.hpp: *       https://github.com/KhronosGroup/OpenCL-CLHPP
src/libprofit/src/profit/cl/cl2.hpp: * reasonable to define C++ bindings for OpenCL.
src/libprofit/src/profit/cl/cl2.hpp: * fixes in the new header as well as additional OpenCL 2.0 features.
src/libprofit/src/profit/cl/cl2.hpp: * Due to the evolution of the underlying OpenCL API the 2.0 C++ bindings
src/libprofit/src/profit/cl/cl2.hpp: * and the range of valid underlying OpenCL runtime versions supported.
src/libprofit/src/profit/cl/cl2.hpp: * The combination of preprocessor macros CL_HPP_TARGET_OPENCL_VERSION and 
src/libprofit/src/profit/cl/cl2.hpp: * CL_HPP_MINIMUM_OPENCL_VERSION control this range. These are three digit
src/libprofit/src/profit/cl/cl2.hpp: * decimal values representing OpenCL runime versions. The default for 
src/libprofit/src/profit/cl/cl2.hpp: * the target is 200, representing OpenCL 2.0 and the minimum is also 
src/libprofit/src/profit/cl/cl2.hpp: * The OpenCL 1.x versions of the C++ bindings included a size_t wrapper
src/libprofit/src/profit/cl/cl2.hpp: * In OpenCL 2.0 OpenCL C is not entirely backward compatibility with 
src/libprofit/src/profit/cl/cl2.hpp: * earlier versions. As a result a flag must be passed to the OpenCL C
src/libprofit/src/profit/cl/cl2.hpp: * compiled to request OpenCL 2.0 compilation of kernels with 1.2 as
src/libprofit/src/profit/cl/cl2.hpp: * For those cases the compilation defaults to OpenCL C 2.0.
src/libprofit/src/profit/cl/cl2.hpp: * - CL_HPP_TARGET_OPENCL_VERSION
src/libprofit/src/profit/cl/cl2.hpp: *   Defines the target OpenCL runtime version to build the header
src/libprofit/src/profit/cl/cl2.hpp: *   against. Defaults to 200, representing OpenCL 2.0.
src/libprofit/src/profit/cl/cl2.hpp: *   Enables device fission for OpenCL 1.2 platforms.
src/libprofit/src/profit/cl/cl2.hpp: *   Default to OpenCL C 1.2 compilation rather than OpenCL C 2.0
src/libprofit/src/profit/cl/cl2.hpp:    #define CL_HPP_TARGET_OPENCL_VERSION 200
src/libprofit/src/profit/cl/cl2.hpp:            if (platver.find("OpenCL 2.") != std::string::npos) {
src/libprofit/src/profit/cl/cl2.hpp:            std::cout << "No OpenCL 2.0 platform found.";
src/libprofit/src/profit/cl/cl2.hpp:#if !defined(CL_HPP_TARGET_OPENCL_VERSION)
src/libprofit/src/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not defined. It will default to 200 (OpenCL 2.0)")
src/libprofit/src/profit/cl/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION != 100 && CL_HPP_TARGET_OPENCL_VERSION != 110 && CL_HPP_TARGET_OPENCL_VERSION != 120 && CL_HPP_TARGET_OPENCL_VERSION != 200
src/libprofit/src/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_TARGET_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 200")
src/libprofit/src/profit/cl/cl2.hpp:# undef CL_HPP_TARGET_OPENCL_VERSION
src/libprofit/src/profit/cl/cl2.hpp:# define CL_HPP_TARGET_OPENCL_VERSION 200
src/libprofit/src/profit/cl/cl2.hpp:#if !defined(CL_HPP_MINIMUM_OPENCL_VERSION)
src/libprofit/src/profit/cl/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION != 100 && CL_HPP_MINIMUM_OPENCL_VERSION != 110 && CL_HPP_MINIMUM_OPENCL_VERSION != 120 && CL_HPP_MINIMUM_OPENCL_VERSION != 200
src/libprofit/src/profit/cl/cl2.hpp:# pragma message("cl2.hpp: CL_HPP_MINIMUM_OPENCL_VERSION is not a valid value (100, 110, 120 or 200). It will be set to 100")
src/libprofit/src/profit/cl/cl2.hpp:# undef CL_HPP_MINIMUM_OPENCL_VERSION
src/libprofit/src/profit/cl/cl2.hpp:# define CL_HPP_MINIMUM_OPENCL_VERSION 100
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION > CL_HPP_TARGET_OPENCL_VERSION
src/libprofit/src/profit/cl/cl2.hpp:# error "CL_HPP_MINIMUM_OPENCL_VERSION must not be greater than CL_HPP_TARGET_OPENCL_VERSION"
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 100 && !defined(CL_USE_DEPRECATED_OPENCL_1_0_APIS)
src/libprofit/src/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_0_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 110 && !defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_1_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 120 && !defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
src/libprofit/src/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_1_2_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION <= 200 && !defined(CL_USE_DEPRECATED_OPENCL_2_0_APIS)
src/libprofit/src/profit/cl/cl2.hpp:# define CL_USE_DEPRECATED_OPENCL_2_0_APIS
src/libprofit/src/profit/cl/cl2.hpp:#include <OpenCL/opencl.h>
src/libprofit/src/profit/cl/cl2.hpp:#include <CL/opencl.h>
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:        *  OpenCL C calls that require arrays of size_t values, whose
src/libprofit/src/profit/cl/cl2.hpp: * \brief The OpenCL C++ bindings are defined within this namespace.
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:    F(cl_device_info, CL_DEVICE_OPENCL_C_VERSION, string) \
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:// Flags deprecated in OpenCL 2.0
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 100 && CL_HPP_MINIMUM_OPENCL_VERSION < 200 && CL_HPP_TARGET_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 110 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION > 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#ifdef CL_DEVICE_GPU_OVERLAP_NV
src/libprofit/src/profit/cl/cl2.hpp:CL_HPP_DECLARE_PARAM_TRAITS_(cl_device_info, CL_DEVICE_GPU_OVERLAP_NV, cl_bool)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp: * OpenCL 1.2 devices do have retain/release.
src/libprofit/src/profit/cl/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp: * OpenCL 1.1 devices do not have retain/release.
src/libprofit/src/profit/cl/cl2.hpp:#endif // ! (CL_HPP_TARGET_OPENCL_VERSION >= 120)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#else // CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:     *  \param devices returns a vector of OpenCL D3D10 devices found. The cl::Device
src/libprofit/src/profit/cl/cl2.hpp:     *  values returned in devices can be used to identify a specific OpenCL
src/libprofit/src/profit/cl/cl2.hpp:     *  The application can query specific capabilities of the OpenCL device(s)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp: * Unload the OpenCL compiler.
src/libprofit/src/profit/cl/cl2.hpp: * \note Deprecated for OpenCL 1.2. Use Platform::unloadCompiler instead.
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:/*! \brief Class interface for creating OpenCL buffers from ID3D10Buffer's.
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
src/libprofit/src/profit/cl/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200 || defined(CL_HPP_USE_CL_IMAGE2D_FROM_BUFFER_KHR)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:    *              The channel order may differ as described in the OpenCL 
src/libprofit/src/profit/cl/cl2.hpp:#endif //#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp: *  \note Deprecated for OpenCL 1.2. Please use ImageGL instead.
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120 && CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:            useCreateImage = (version >= 0x10002); // OpenCL 1.2 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif  // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 120
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp: * was performed by OpenCL anyway.
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:     * \param context A valid OpenCL context in which to construct the program.
src/libprofit/src/profit/cl/cl2.hpp:     * \param devices A vector of OpenCL device objects for which the program will be created.
src/libprofit/src/profit/cl/cl2.hpp:     *   CL_INVALID_DEVICE if OpenCL devices listed in devices are not in the list of devices associated with context.
src/libprofit/src/profit/cl/cl2.hpp:     *   CL_OUT_OF_HOST_MEMORY if there is a failure to allocate resources required by the OpenCL implementation on the host.
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:                useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:               useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200 && CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:            useWithProperties = (version >= 0x20000); // OpenCL 2.0 or above
src/libprofit/src/profit/cl/cl2.hpp:#elif CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_MINIMUM_OPENCL_VERSION < 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#else // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:     *     The pattern type must be an accepted OpenCL data type.
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp:     * Enqueues a command that will release a coarse-grained SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if defined(CL_USE_DEPRECATED_OPENCL_1_2_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 120
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if defined(CL_USE_DEPRECATED_OPENCL_1_1_APIS)
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_USE_DEPRECATED_OPENCL_1_1_APIS
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp: * SVM buffer back to the OpenCL runtime.
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#endif // CL_HPP_TARGET_OPENCL_VERSION >= 110
src/libprofit/src/profit/cl/cl2.hpp:#if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/cl2.hpp:#endif // #if CL_HPP_TARGET_OPENCL_VERSION >= 200
src/libprofit/src/profit/cl/moffat-double.cl: * Double-precision Moffat profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/moffat-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/moffat-float.cl: * Single-precision Moffat profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/moffat-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/moffat-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/common-double.cl: * Common double-precision OpenCL routines for libprofit
src/libprofit/src/profit/cl/common-double.cl:#if __OPENCL_C_VERSION__ < 120
src/libprofit/src/profit/cl/common-double.cl:#pragma OPENCL EXTENSION cl_khr_fp64: enable
src/libprofit/src/profit/cl/common-float.cl: * Common single-precision OpenCL routines for libprofit
src/libprofit/src/profit/cl/sersic-float.cl: * Single-precision Sersic profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/sersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-float.cl: * Single-precision King profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/sersic-double.cl: * Double-precision Sersic profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/sersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/sersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/ferrer-double.cl: * Double-precision Ferrer profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/ferrer-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/ferrer-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-double.cl: * Double-precision King profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/king-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/king-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/convolve-double.cl: * double 2D convolution OpenCL implementation for libprofit
src/libprofit/src/profit/cl/brokenexponential-float.cl: * Single-precision Broken Exponential profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/brokenexponential-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/brokenexponential-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-float.cl: * Single-precision Core-Sersic profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-float.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-float.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-double.cl: * Double-precision Core-Sersic profile OpenCL kernel implementation for libprofit
src/libprofit/src/profit/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/cl/coresersic-double.cl:#if __OPENCL_C_VERSION__ <= 120
src/libprofit/src/profit/cl/coresersic-double.cl:#endif /* __OPENCL_C_VERSION__ */
src/libprofit/src/profit/library.h:/// Returns whether libprofit was compiled with OpenCL support
src/libprofit/src/profit/library.h:/// @return Whether libprofit was compiled with OpenCL support
src/libprofit/src/profit/library.h:PROFIT_API bool has_opencl();
src/libprofit/src/profit/library.h:/// If OpenCL is supported, returns the major portion of the highest OpenCL
src/libprofit/src/profit/library.h:/// compiled against a platform supporting OpenCL 2.1, this method returns 2.
src/libprofit/src/profit/library.h:/// If OpenCL is not supported, the result is undefined.
src/libprofit/src/profit/library.h:/// @return The major highest OpenCL platform version that libprofit can work
src/libprofit/src/profit/library.h:PROFIT_API unsigned short opencl_version_major();
src/libprofit/src/profit/library.h:/// If OpenCL is supported, returns the minor portion of the highest OpenCL
src/libprofit/src/profit/library.h:/// compiled against a platform supporting OpenCL 1.2, this method returns 2.
src/libprofit/src/profit/library.h:/// If OpenCL is not supported, the result is undefined.
src/libprofit/src/profit/library.h:PROFIT_API unsigned short opencl_version_minor();
src/libprofit/src/profit/sersic.cpp:#include "profit/opencl.h"
src/libprofit/src/profit/sersic.cpp:#ifdef PROFIT_OPENCL
src/libprofit/src/profit/sersic.cpp:#endif /* PROFIT_OPENCL */
src/libprofit/src/profit/model.cpp:	opencl_env(),
src/libprofit/src/profit/model.cpp:	opencl_env(),
src/Makevars.in:          libprofit/src/profit/opencl.o \
src/Makevars.win:          libprofit/src/profit/opencl.o \
src/Makevars.win:	sed 's/@PROFIT_VERSION_MAJOR@/1/; s/@PROFIT_VERSION_MINOR@/7/; s/@PROFIT_VERSION_PATCH@/3/; s/@PROFIT_VERSION_SUFFIX@//; s/#cmakedefine PROFIT_USES_R/#define PROFIT_USES_R/; s/#cmakedefine PROFIT_USES_GSL/#undef PROFIT_USES_GSL/; s/#cmakedefine PROFIT_DEBUG/#undef PROFIT_DEBUG/; s/#cmakedefine PROFIT_OPENCL/#undef PROFIT_OPENCL/; s/#cmakedefine PROFIT_OPENMP/#undef PROFIT_OPENMP/; s/#cmakedefine PROFIT_FFTW/#undef PROFIT_FFTW/; s/#cmakedefine PROFIT_FFTW_OPENMP/#undef PROFIT_FFTW_OPENMP/; s/#cmakedefine PROFIT_HAS_SSE2/#undef PROFIT_HAS_SSE2/; s/#cmakedefine PROFIT_HAS_AVX/#undef PROFIT_HAS_AVX/; s/@PROFIT_OPENCL_MAJOR@//; s/@PROFIT_OPENCL_MINOR@//; s/@PROFIT_OPENCL_TARGET_VERSION@//' libprofit/src/profit/config.h.in > libprofit/src/profit/config.h
src/r_profit.cpp: * OpenCL-related functionality follows
src/r_profit.cpp:SEXP _R_profit_openclenv_info() {
src/r_profit.cpp:	map<int, OpenCL_plat_info> clinfo;
src/r_profit.cpp:		clinfo = get_opencl_info();
src/r_profit.cpp:		os << "Error while querying OpenCL environment: " << e.what();
src/r_profit.cpp:	SET_STRING_ELT(r_platinfo_names, 1, Rf_mkChar("opencl_version"));
src/r_profit.cpp:		SEXP r_plat_clver = PROTECT(Rf_ScalarReal(plat_info.supported_opencl_version/100.));
src/r_profit.cpp:struct openclenv_wrapper {
src/r_profit.cpp:	OpenCLEnvPtr env;
src/r_profit.cpp:void _R_profit_openclenv_finalizer(SEXP ptr) {
src/r_profit.cpp:	openclenv_wrapper *wrapper = reinterpret_cast<openclenv_wrapper *>(R_ExternalPtrAddr(ptr));
src/r_profit.cpp:OpenCLEnvPtr unwrap_openclenv(SEXP openclenv) {
src/r_profit.cpp:	if( TYPEOF(openclenv) != EXTPTRSXP ) {
src/r_profit.cpp:		Rf_error("Given openclenv not of proper type\n");
src/r_profit.cpp:	openclenv_wrapper *wrapper = reinterpret_cast<openclenv_wrapper *>(R_ExternalPtrAddr(openclenv));
src/r_profit.cpp:		Rf_error("No OpenCL environment found in openclenv\n");
src/r_profit.cpp:SEXP _R_profit_openclenv(SEXP plat_idx, SEXP dev_idx, SEXP use_dbl) {
src/r_profit.cpp:	OpenCLEnvPtr env;
src/r_profit.cpp:		env = get_opencl_environment(platform_idx, device_idx, use_double, false);
src/r_profit.cpp:	} catch (const opencl_error &e) {
src/r_profit.cpp:		os << "Error while creating OpenCL environment for plat/dev/double " << platform_idx << "/" << device_idx << "/" << use_double << ": " << e.what();
src/r_profit.cpp:		os << "Error while creating OpenCL environment, invalid parameter: " << e.what();
src/r_profit.cpp:	openclenv_wrapper *wrapper = new openclenv_wrapper();
src/r_profit.cpp:	SEXP r_openclenv = R_MakeExternalPtr(wrapper, Rf_install("OpenCL_env"), R_NilValue);
src/r_profit.cpp:	PROTECT(r_openclenv);
src/r_profit.cpp:	R_RegisterCFinalizerEx(r_openclenv, _R_profit_openclenv_finalizer, TRUE);
src/r_profit.cpp:	return r_openclenv;
src/r_profit.cpp:	if (profit::has_opencl()) {
src/r_profit.cpp:		convolvers.push_back("opencl");
src/r_profit.cpp:                              SEXP openclenv)
src/r_profit.cpp:	if (openclenv != R_NilValue ) {
src/r_profit.cpp:		if((conv_prefs.opencl_env = unwrap_openclenv(openclenv)) == nullptr) {
src/r_profit.cpp:	/* OpenCL environment, if any */
src/r_profit.cpp:	SEXP openclenv = _get_list_element(model_list, "openclenv");
src/r_profit.cpp:	if( openclenv != R_NilValue ) {
src/r_profit.cpp:		if( TYPEOF(openclenv) != EXTPTRSXP ) {
src/r_profit.cpp:			Rf_error("Given openclenv not of proper type\n");
src/r_profit.cpp:		openclenv_wrapper *wrapper = reinterpret_cast<openclenv_wrapper *>(R_ExternalPtrAddr(openclenv));
src/r_profit.cpp:			Rf_error("No OpenCL environment found in openclenv\n");
src/r_profit.cpp:		m.set_opencl_env(wrapper->env);
src/r_profit.cpp:	                             SEXP openclenv) {
src/r_profit.cpp:		                                fft_effort, omp_threads, openclenv);
src/r_profit.cpp:	SEXP R_profit_openclenv_info() {
src/r_profit.cpp:		return _R_profit_openclenv_info();
src/r_profit.cpp:	SEXP R_profit_openclenv(SEXP plat_idx, SEXP dev_idx, SEXP use_dbl) {
src/r_profit.cpp:		return _R_profit_openclenv(plat_idx, dev_idx, use_dbl);
src/r_profit.cpp:		{"R_profit_openclenv_info", (DL_FUNC) &R_profit_openclenv_info, 0},
src/r_profit.cpp:		{"R_profit_openclenv",      (DL_FUNC) &R_profit_openclenv,      3},

```
