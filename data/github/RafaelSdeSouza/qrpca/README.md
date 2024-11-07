# https://github.com/RafaelSdeSouza/qrpca

```console
R/qrpca.R:#'  @param cuda a logical value indicating whether cuda acceleration should be
R/qrpca.R:qrpca <- function(x,center = TRUE, scale = FALSE,cuda = FALSE){
R/qrpca.R:  if(cuda == TRUE){device = torch_device("cuda:0")} else
README.md:qrpca behaves similarly prcomp. But employs a QR-based PCA instead of applying singular value decomposition on the original matrix. The code uses torch under the hood for matrix operations and supports GPU acceleration.
README.md:  system.time(qrpca(X,cuda = TRUE))
man/qrpca.Rd:qrpca(x, center = TRUE, scale = FALSE, cuda = FALSE)
man/qrpca.Rd:@param cuda a logical value indicating whether cuda acceleration should be

```
