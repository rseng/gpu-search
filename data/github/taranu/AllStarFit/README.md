# https://github.com/taranu/AllStarFit

```console
R/allStarFit.R:		#plotSourceFits(psfits, procmaps, band)
R/allStarFit.R:allStarFitPSF <- function(procmaps, psfits, bands = names(procmaps$single), maxcomp=1,
R/allStarFit.R:	stopifnot(all(bands %in% names(procmaps$maps)))
R/allStarFit.R:	stopifnot(all(bands %in% names(procmaps$proc$single)))
R/allStarFit.R:	segim = procmaps$proc$multi$proc$segim
R/allStarFit.R:	mask = procmaps$proc$multi$mask
R/allStarFit.R:	#skypix = procmaps$proc$multi$proc$objects_redo==0
R/allStarFit.R:	skypix = procmaps$proc$multi$proc$segim<=0
R/allStarFit.R:		bproc = procmaps$proc$single[[band]]
R/allStarFit.R:		bmaps = procmaps$maps[[band]]
R/allStarFit.R:allStarFitExtended <- function(procmaps, psffits,
R/allStarFit.R:	bands = names(procmaps$maps), maxcomp=2)
R/allStarFit.R:	stopifnot(all(bands %in% names(procmaps$maps)))
R/allStarFit.R:	stopifnot(all(bands %in% names(procmaps$proc$single)))
R/allStarFit.R:		bmaps = procmaps$maps[[band]]
R/allStarFit.R:		sources = procmaps$single[[band]]$sources$extended
R/allStarFit.R:		fits[[band]] = fitGalaxies(process = procmaps$proc, psffit = bpsffit,
R/allStarFit.R:plotSourceFits <- function(bpsfits, procmaps, band, fullfits=list(NULL,NULL),
R/allStarFit.R:	bmaps = procmaps$maps[[band]]
R/allStarFit.R:	bproc = procmaps$proc$single[[band]]
R/profitModelQuantiles.R:	finesample=1L, nopsf=TRUE, openclenv=NULL)
R/profitModelQuantiles.R:		finesample = finesample, returnfine = TRUE, openclenv=openclenv)$z
R/profitModelQuantiles.R:	modelpars = NULL, minfluxfrac=0.95, openclenv=NULL)
R/profitModelQuantiles.R:	if(!is.null(openclenv) && identical(typeof(openclenv),"externalptr") &&
R/profitModelQuantiles.R:		 !identical(openclenv, new("externalptr"))) fitdata$Data$openclenv = openclenv
R/profitModelQuantiles.R:	else if(identical(fitdata$Data$openclenv, new("externalptr"))) fitdata$Data$openclenv = NULL
R/profitModelQuantiles.R:	finemodel = profitMakeModelFine(best, fitdata$Data, openclenv = fitdata$Data$openclenv)
R/profitModelQuantiles.R:			openclenv = fitdata$Data$openclenv)
R/profitModelQuantiles.R:		dim = kindim, remax = 1, openclenv = fitdata$Data$openclenv)$z)
R/profitModelQuantiles.R:		dim = kindim, openclenv = fitdata$Data$openclenv)$z
R/profitModelQuantiles.R:		openclenv = fitdata$Data$openclenv)$image/gain
R/profit.R:	galfit=NULL, maxwalltime=Inf, openclenvs = profitGetOpenCLEnvs(make.envs=TRUE),
R/profit.R:	if(profitHasOpenCL()) convmethods = c(convmethods,"opencl")
R/profit.R:		benchopenclenvs = openclenvs
R/profit.R:				benchopenclenvs = openclenvs
R/profit.R:				benchopenclenvs = openclenvs)
R/profit.R:					benchopenclenvs = openclenvs)
R/profit.R:						benchopenclenvs = openclenvs)
R/profit.R:							benchopenclenvs = openclenvs)
R/profit.R:						benchintmethods = profitAvailableIntegrators(), benchopenclenvs = openclenvs,
R/profit.R:						constraints = constraints, benchopenclenvs = openclenvs)
R/profit.R:						constraints = newData$constraints, benchopenclenvs = openclenvs)
R/profit.R:				benchopenclenvs = openclenvs)
R/profit.R:					benchopenclenvs = openclenvs)
R/profit.R:							benchopenclenvs = openclenvs)

```
