# https://github.com/ShixiangWang/sigminer

```console
R/show_sig_fit.R:#' @inheritParams ggpubr::ggboxplot
R/show_sig_fit.R:#' @inheritParams ggpubr::ggpar
R/show_sig_fit.R:    boxplot = ggpubr::ggboxplot,
R/show_sig_fit.R:    violin = ggpubr::ggviolin,
R/show_sig_fit.R:    scatter = ggpubr::ggscatter
R/show_sig_bootstrap.R:#' @inheritParams ggpubr::ggboxplot
R/show_sig_bootstrap.R:#' @inheritParams ggpubr::ggpar
R/show_sig_bootstrap.R:#' @param ... other parameters passing to [ggpubr::ggboxplot] or [ggpubr::ggviolin].
R/show_sig_bootstrap.R:    boxplot = ggpubr::ggboxplot,
R/show_sig_bootstrap.R:    violin = ggpubr::ggviolin
R/show_sig_bootstrap.R:    boxplot = ggpubr::ggboxplot,
R/show_sig_bootstrap.R:    violin = ggpubr::ggviolin
R/show_sig_bootstrap.R:    boxplot = ggpubr::ggboxplot,
R/show_sig_bootstrap.R:    violin = ggpubr::ggviolin
R/show_group_comparison.R:#' @param ... other paramters pass to [ggpubr::compare_means()] or [ggpubr::stat_compare_means()]
R/show_group_comparison.R:        if (!requireNamespace("ggpubr", quietly = TRUE)) {
R/show_group_comparison.R:          stop("'ggpubr' package is needed for plotting p values.")
R/show_group_comparison.R:          p <- p + ggpubr::stat_compare_means(method = method, ...)
R/show_group_comparison.R:          # p <- p + ggpubr::stat_compare_means(
R/show_group_comparison.R:          p <- p + ggpubr::stat_pvalue_manual(p_df, label = "p.adj")
R/get_adj_p.R:#' Setting `aes(label=..p.adj..)` in [ggpubr::compare_means()] does not
R/get_adj_p.R:#' show adjust p values. The returned result of this function can be combined with [ggpubr::stat_pvalue_manual()] to fix
R/get_adj_p.R:#' More info see [ggpubr::compare_means()], [ggpubr::stat_compare_means()] and [stats::p.adjust()].
R/get_adj_p.R:#' @param ... other arguments passed to [ggpubr::compare_means()]
R/get_adj_p.R:#' @source https://github.com/kassambara/ggpubr/issues/143
R/get_adj_p.R:#' library(ggpubr)
R/get_adj_p.R:#' # proposed by author of ggpubr
R/get_adj_p.R:  pvalues <- ggpubr::compare_means(
R/output.R:    p1 <- show_sig_fit(expo, palette = NULL, plot_fun = "boxplot") + ggpubr::rotate_x_text()
R/output.R:    p2 <- show_sig_fit(expo, palette = NULL, plot_fun = "violin") + ggpubr::rotate_x_text()
R/output.R:    p3 <- show_sig_fit(rel_expo, palette = NULL, plot_fun = "boxplot") + ggpubr::rotate_x_text()
R/output.R:    p4 <- show_sig_fit(rel_expo, palette = NULL, plot_fun = "violin") + ggpubr::rotate_x_text()
R/output.R:    p1 <- show_sig_fit(expo, palette = NULL, plot_fun = "boxplot", signatures = sigs) + ggpubr::rotate_x_text()
R/output.R:    p2 <- show_sig_fit(expo, palette = NULL, plot_fun = "violin", signatures = sigs) + ggpubr::rotate_x_text()
R/output.R:    p3 <- show_sig_fit(rel_expo, palette = NULL, plot_fun = "boxplot", signatures = sigs) + ggpubr::rotate_x_text()
R/output.R:    p4 <- show_sig_fit(rel_expo, palette = NULL, plot_fun = "violin", signatures = sigs) + ggpubr::rotate_x_text()
R/output.R:    p1 <- show_sig_bootstrap_stability(x) + theme(legend.position = "none") + ggpubr::rotate_x_text()
R/output.R:    p2 <- show_sig_bootstrap_exposure(x) + theme(legend.position = "none") + ggpubr::rotate_x_text()
R/output.R:    p1 <- show_sig_bootstrap_stability(x, signatures = sigs) + theme(legend.position = "none") + ggpubr::rotate_x_text()
R/output.R:    p2 <- show_sig_bootstrap_exposure(x, signatures = sigs) + theme(legend.position = "none") + ggpubr::rotate_x_text()
R/output.R:      p <- show_sig_bootstrap_exposure(x, signatures = sigs, sample = i) + theme(legend.position = "none") + ggpubr::rotate_x_text()
R/output.R:      p <- show_sig_bootstrap_exposure(x, sample = i) + theme(legend.position = "none") + ggpubr::rotate_x_text()
README.Rmd:-   elegant plots powered by R packages **ggplot2**, **ggpubr**, **cowplot** and **patchwork**.
tests/testthat/test-roxytest-testexamples-get_adj_p.R:  library(ggpubr)
tests/testthat/test-roxytest-testexamples-get_adj_p.R:  # proposed by author of ggpubr
README.md:- elegant plots powered by R packages **ggplot2**, **ggpubr**,
DESCRIPTION:    ggpubr,
man/show_sig_fit.Rd:\code{\link[ggplot2]{geom_boxplot}}, \code{\link[ggpubr]{ggpar}} and
man/show_sig_fit.Rd:\code{\link[ggpubr]{facet}}.}
man/get_adj_p.Rd:https://github.com/kassambara/ggpubr/issues/143
man/get_adj_p.Rd:\item{...}{other arguments passed to \code{\link[ggpubr:compare_means]{ggpubr::compare_means()}}}
man/get_adj_p.Rd:Setting \code{aes(label=..p.adj..)} in \code{\link[ggpubr:compare_means]{ggpubr::compare_means()}} does not
man/get_adj_p.Rd:show adjust p values. The returned result of this function can be combined with \code{\link[ggpubr:stat_pvalue_manual]{ggpubr::stat_pvalue_manual()}} to fix
man/get_adj_p.Rd:More info see \code{\link[ggpubr:compare_means]{ggpubr::compare_means()}}, \code{\link[ggpubr:stat_compare_means]{ggpubr::stat_compare_means()}} and \code{\link[stats:p.adjust]{stats::p.adjust()}}.
man/get_adj_p.Rd:library(ggpubr)
man/get_adj_p.Rd:# proposed by author of ggpubr
man/show_group_comparison.Rd:\item{...}{other paramters pass to \code{\link[ggpubr:compare_means]{ggpubr::compare_means()}} or \code{\link[ggpubr:stat_compare_means]{ggpubr::stat_compare_means()}}
man/show_sig_bootstrap.Rd:\item{...}{other parameters passing to \link[ggpubr:ggboxplot]{ggpubr::ggboxplot} or \link[ggpubr:ggviolin]{ggpubr::ggviolin}.}

```
