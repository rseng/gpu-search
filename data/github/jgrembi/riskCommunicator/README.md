# https://github.com/jgrembi/riskCommunicator

```console
R/plot.gComp.R:#' @importFrom ggpubr ggarrange annotate_figure text_grob
R/plot.gComp.R:  plot <- ggpubr::ggarrange(hist, qqplot, ncol = 2)
R/plot.gComp.R:  ggpubr::annotate_figure(plot,
R/plot.gComp.R:                          bottom = ggpubr::text_grob(note, color = "blue", size = 18))
vignettes/Vignette_manuscript.Rmd:library(ggpubr)
DESCRIPTION:    ggpubr,
NAMESPACE:importFrom(ggpubr,annotate_figure)
NAMESPACE:importFrom(ggpubr,ggarrange)
NAMESPACE:importFrom(ggpubr,text_grob)

```
