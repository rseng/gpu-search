# https://github.com/eggplantbren/DNest3

```console
include/MTSampler.h:		double logPush(int index) const;
include/MTSamplerImpl.h:	logA += logPush(proposedIndex) - logPush(indices[thread][which]);
include/MTSamplerImpl.h:double MTSampler<ModelType>::logPush(int index) const
include/MTSamplerImpl.h:	double max_logPush = -1E300;
include/MTSamplerImpl.h:			if(logPush(indices[i][j]) > max_logPush)
include/MTSamplerImpl.h:				max_logPush = logPush(indices[i][j]);
include/MTSamplerImpl.h:			if(logPush(indices[i][j]) < -5.
include/MTSamplerImpl.h:					}while(!good[iCopy][jCopy] || randomU() >= exp(logPush(indices[i][j]) - max_logPush));
include/Sampler.h:		double logPush(int index) const;
include/SamplerImpl.h:	logA += logPush(proposedIndex) - logPush(indices[which]);
include/SamplerImpl.h:double Sampler<ModelType>::logPush(int index) const
include/SamplerImpl.h:		if(logPush(indices[i]) < -5.)

```
