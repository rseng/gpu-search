# https://github.com/mmollina/viewpoly

```console
Dockerfile:RUN Rscript -e 'remotes::install_version("ggpubr",upgrade="never", version = "0.4.0")'
R/mod_qtl_view.R:#' @importFrom ggpubr ggarrange
tests/testthat/test-QTLpoly.R:    # library(ggpubr)
DESCRIPTION:    ggpubr,
NAMESPACE:importFrom(ggpubr,ggarrange)

```
