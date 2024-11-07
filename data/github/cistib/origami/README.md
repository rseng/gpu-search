# https://github.com/cistib/origami

```console
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:%   C = CONVNFFT(..., SHAPE, DIMS, GPU)
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:%       GPU is boolean flag, see next
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:%       - 'GPU', boolean. If GPU is TRUE Jacket/GPU FFT engine will be used
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:%       By default GPU is FALSE.
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:%       02-Sep-2009: GPU/JACKET option
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:elseif ~isstruct(options) % GPU options
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:    options = struct('GPU', options);
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:% GPU enable flag
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:GPU = getoption(options, 'GPU', false);
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:GPU = GPU && ~isempty(which('ginfo'));
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:if GPU % GPU/Jacket FFT
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:        % We need to swap dimensions because GPU FFT works along the
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:if GPU
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:if GPU % GPU/Jacket FFT
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:% GPU/Jacket
SyntheticData/functions/CONVNFFT_Folder/convnfft.m:if GPU

```
