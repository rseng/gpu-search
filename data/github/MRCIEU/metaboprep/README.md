# https://github.com/MRCIEU/metaboprep

```console
inst/rmarkdown/metaboprep_Report.Rmd:ggpubr::ggtexttable(tout, rows = NULL, theme = ggpubr::ttheme("mBlue") )
inst/rmarkdown/metaboprep_Report.Rmd:p = ggpubr::ggarrange(  r_mis[[4]][[1]] ,
inst/rmarkdown/metaboprep_Report.Rmd:ggpubr::annotate_figure(p,top = paste0("-- Initial raw data set: Estimates of missingness for samples and features --\n") )
inst/rmarkdown/metaboprep_Report.Rmd:ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mBlue"))
inst/rmarkdown/metaboprep_Report.Rmd:ptable = ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mBlue"))
inst/rmarkdown/metaboprep_Report.Rmd:ptable = ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mGreen"))
inst/rmarkdown/metaboprep_Report.Rmd:ggpubr::ggarrange(p1, p2, labels = c("A", "B"), ncol = 2, nrow = 1)
inst/rmarkdown/metaboprep_Report_v0.Rmd:ggpubr::ggtexttable(tout, rows = NULL, theme = ggpubr::ttheme("mBlue") )
inst/rmarkdown/metaboprep_Report_v0.Rmd:p = ggpubr::ggarrange(  r_mis[[4]][[1]] ,
inst/rmarkdown/metaboprep_Report_v0.Rmd:ggpubr::annotate_figure(p,top = paste0("-- Initial raw data set: Estimates of missingness for samples and features --\n") )
inst/rmarkdown/metaboprep_Report_v0.Rmd:ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mBlue"))
inst/rmarkdown/metaboprep_Report_v0.Rmd:ptable = ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mBlue"))
inst/rmarkdown/metaboprep_Report_v0.Rmd:ptable = ggpubr::ggtexttable( temp, rows = NULL, theme = ggpubr::ttheme("mGreen"))
inst/rmarkdown/metaboprep_Report_v0.Rmd:ggpubr::ggtexttable( outvals, rows = NULL, theme = ggpubr::ttheme("mBlue"))
inst/rmarkdown/metaboprep_Report_v0.Rmd:ggpubr::ggarrange(p1, p2, labels = c("A", "B"), ncol = 2, nrow = 1)
R/feature_plots.R:#' @importFrom ggpubr ggarrange ggtexttable ggexport
R/feature_plots.R:  pkgs = c("ggpubr", "RColorBrewer", "magrittr", "ggplot2")
R/feature_plots.R:    p3 = ggpubr::ggtexttable(ss)
R/feature_plots.R:    top = ggpubr::ggarrange(p1,p2, nrow = 1) 
R/feature_plots.R:    out = ggpubr::ggarrange(  top, p3, nrow = 2, heights = c(4,1) )
R/feature_plots.R:  ggpubr::ggarrange(plotlist = plotsout, ncol = 1, nrow = 3) %>% 
R/feature_plots.R:    ggpubr::ggexport(filename = f, width = 13, height = 15)
R/multivariate.anova.R:#' @importFrom ggpubr ggtexttable ttheme
R/multivariate.anova.R:  pkgs = c("stats", "car", "tibble", "dplyr", "ggpubr", "magrittr")
R/multivariate.anova.R:  outtable <- ggpubr::ggtexttable(outtable, rows = NULL, 
R/multivariate.anova.R:                          theme = ggpubr::ttheme("mBlue"))
R/outlier.summary.R:#' @importFrom ggpubr ggtexttable ttheme
R/outlier.summary.R:  pkgs = c("psych", "ggpubr")
R/outlier.summary.R:  outtable <- ggpubr::ggtexttable(outtable, rows = NULL, 
R/outlier.summary.R:                          theme = ggpubr::ttheme("mBlue"))
R/missingness.sum.R:#' @importFrom ggpubr ggtexttable ttheme
R/missingness.sum.R:#' ggpubr::ggarrange(plotlist = ms$plotsout, ncol = 2, nrow = 2)
R/missingness.sum.R:  pkgs = c("RColorBrewer", "magrittr", "tibble", "ggpubr", "ggplot2")
R/missingness.sum.R:  sumtable <- ggpubr::ggtexttable(missingness_table, rows = NULL, 
R/missingness.sum.R:                          theme = ggpubr::ttheme("mBlue"))
R/missingness.sum.R:  miss_samplesize_table <- ggpubr::ggtexttable(fmiss_samplesize, rows = NULL, 
R/missingness.sum.R:                                       theme = ggpubr::ttheme("mBlue"))
R/missingness.sum.R:  miss_samplesize_table <- ggpubr::ggtexttable(fmiss_samplesize, rows = NULL, 
R/missingness.sum.R:                                       theme = ggpubr::ttheme("mBlue"))
DESCRIPTION:    ggpubr,
NAMESPACE:importFrom(ggpubr,ggarrange)
NAMESPACE:importFrom(ggpubr,ggexport)
NAMESPACE:importFrom(ggpubr,ggtexttable)
NAMESPACE:importFrom(ggpubr,ttheme)
man/missingness.sum.Rd:ggpubr::ggarrange(plotlist = ms$plotsout, ncol = 2, nrow = 2)

```
