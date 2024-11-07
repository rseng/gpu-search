# https://github.com/pratheesh3780/grapesAgri1

```console
inst/desc/report.Rmd:library(ggpubr)
inst/desc/app.R:library(ggpubr)
inst/desc/app.R:            ggpubr::ggqqplot(csvfile()[, input$variable], color = input$color)
inst/desc/app.R:        ggpubr::ggqqplot(csvfile()[, input$variable], color = input$color)
inst/comp_mean/app.R:library(ggpubr)
inst/comp_mean/app.R:            ggpubr::ggboxplot(
inst/crd/app.R:library(ggpubr)
inst/REFERENCES.bib:  @Manual{ggpubr_2020,
inst/REFERENCES.bib:    title = {ggpubr: 'ggplot2' Based Publication Ready Plots},
inst/REFERENCES.bib:    url = {https://CRAN.R-project.org/package=ggpubr},
R/rbd.R:#' \code{ggboxplot} function of \code{ggpubr} package is used for
R/rbd.R:#' \insertRef{ggpubr_2020}{grapesAgri1}
R/crd.R:#' \code{ggboxplot} function of \code{ggpubr} package is used for
R/crd.R:#' \insertRef{ggpubr_2020}{grapesAgri1}
R/desc.R:#' to obtain histogram and boxplot respectively. \code{ggqqplot} of package \code{ggpubr} (Alboukadel Kassambara,2020)
R/desc.R:#' \insertRef{ggpubr_2020}{grapesAgri1}
R/ttest.R:#' of \code{ggpubr} package is used to draw boxplot. Paired plot is obtained
R/ttest.R:#' \insertRef{ggpubr_2020}{grapesAgri1}
joss/paper.bib:@manual{ggpubr_2020,
joss/paper.bib:  title = {ggpubr: 'ggplot2' Based Publication Ready Plots},
joss/paper.bib:  url = {https://CRAN.R-project.org/package=ggpubr}
joss/paper.md:DescApp() function uses `descr` and `stby` functions of `summarytools` package [@Dominic_Comtois_2021] to calculate summary statistics and summary statistics by group. `knitr` [@Yihui_Xie_2021] and `kableExtra`[@Hao_zhu_2021] packages were used to produce HTML tables. `shapiro.test`, `qqnorm` and `qqline` functions of `stats` package were used for the Test of Homogeneity of variance and obtaining Q-Q plot. `hist` and `boxplot` of package `graphics` were used to obtain histogram and boxplot respectively. `ggqqplot` of package `ggpubr `[@ggpubr_2020] is also used to plot Q-Q plot in the app.
joss/paper.md:ttApp()function uses `t.test` function to calculate t statistic. Descriptive statistics were calculated using `stat.desc` function of `pastecs` package. `var.test` function is used for F-test. `ggboxplot` function of `ggpubr` [@ggpubr_2020] package is used to draw boxplot. Paired plot is obtained using `paired` function of package `PairedData`[@paired_data2018].
joss/paper.md:crdApp() uses `anova` function of `stats` package to obtain one-way ANOVA. `LSD.test`,`duncan.test` and `HSD.test` functions of `agricolae` [@agricolae_2020] package is used for multiple comparison test like LSD,DMRT and Tukey respectively. `ggboxplot` function of `ggpubr` [@ggpubr_2020] package is used for boxplot. `ggplot` function of `ggplot2`[@ggplot_2016] is used for barchart with confidence interval.
joss/paper.md:rbdApp() uses `anova` function of `stats` package to obtain two-way ANOVA. `LSD.test`,`duncan.test` and `HSD.test` functions of `agricolae` package [@agricolae_2020] is used for multiple comparison test like LSD,DMRT and Tukey respectively. `ggboxplot` function of `ggpubr` package [@ggpubr_2020] is used for boxplot. `ggplot` function of `ggplot2` [@ggplot_2016] is used for barchart with confidence interval.
DESCRIPTION:        ggpubr(>= 0.4.0),
man/descApp.Rd:to obtain histogram and boxplot respectively. \code{ggqqplot} of package \code{ggpubr} (Alboukadel Kassambara,2020)
man/descApp.Rd:\insertRef{ggpubr_2020}{grapesAgri1}
man/rbdApp.Rd:\code{ggboxplot} function of \code{ggpubr} package is used for
man/rbdApp.Rd:\insertRef{ggpubr_2020}{grapesAgri1}
man/ttApp.Rd:of \code{ggpubr} package is used to draw boxplot. Paired plot is obtained
man/ttApp.Rd:\insertRef{ggpubr_2020}{grapesAgri1}
man/crdApp.Rd:\code{ggboxplot} function of \code{ggpubr} package is used for
man/crdApp.Rd:\insertRef{ggpubr_2020}{grapesAgri1}

```
