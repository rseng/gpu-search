# https://github.com/SofieVG/FlowSOM

```console
R/0_FlowSOM.R:#' @importFrom ggpubr annotate_figure ggarrange text_grob
R/0_FlowSOM.R:    p <- ggpubr::annotate_figure(ggarrange(plotlist = plots_list,
R/0_FlowSOM.R:                                 bottom = ggpubr::text_grob("Files"))
R/2_buildSOM.R:#' p <- ggpubr::ggarrange(plotlist = c(list(gr_1$tree), gr_2),
R/6_deprecated.R:#' p <- ggpubr::ggarrange(plotlist = c(list(gr_1$tree), gr_2),
R/5_plotFunctions.R:#' @importFrom ggpubr get_legend ggarrange
R/5_plotFunctions.R:  l2 <- ggpubr::get_legend(p)
R/5_plotFunctions.R:    l2 <- ggpubr::as_ggplot(l2)
R/5_plotFunctions.R:    p <- ggpubr::ggarrange(p,
R/5_plotFunctions.R:                           ggpubr::ggarrange(l1, l2,
R/5_plotFunctions.R:  p <- ggpubr::ggarrange(plotlist = plotList, common.legend = TRUE, 
R/5_plotFunctions.R:#' @importFrom ggpubr ggarrange
R/5_plotFunctions.R:    print(ggpubr::ggarrange(plotlist = plots_list, 
R/5_plotFunctions.R:#' @importFrom ggpubr ggarrange
R/5_plotFunctions.R:    p <- ggpubr::ggarrange(fP$tree, 
R/5_plotFunctions.R:                           ggpubr::ggarrange(l, fP$backgroundLegend, ncol = 1), 
R/5_plotFunctions.R:#' @importFrom ggpubr ggarrange ttheme ggtexttable
R/5_plotFunctions.R:      ggpubr::ggarrange(p2.1, p2.2, 
R/5_plotFunctions.R:  t1 <- ggpubr::ggtexttable(t(datatable1), theme = ggpubr::ttheme("minimal"))
R/5_plotFunctions.R:        print(ggpubr::ggtexttable(table2, theme = ggpubr::ttheme("minimal"), 
R/5_plotFunctions.R:      print(ggpubr::ggtexttable(table3, theme = ggpubr::ttheme("minimal"), 
R/5_plotFunctions.R:#' @importFrom ggpubr ggarrange
R/5_plotFunctions.R:    p <- ggpubr::ggarrange(p, ggpubr::ggarrange(l1, l2, ncol = 1), NULL,
R/5_plotFunctions.R:#' @importFrom ggpubr ggarrange
R/5_plotFunctions.R:  p <- suppressMessages(ggpubr::ggarrange(plotlist = plotList))
R/3_buildMST.R:#' @importFrom ggpubr ggarrange
vignettes/FlowSOM.Rnw:p <- ggpubr::ggarrange(plotlist = list(gr_1$tree, gr_2$tree, 
DESCRIPTION:    ggpubr,
NAMESPACE:importFrom(ggpubr,annotate_figure)
NAMESPACE:importFrom(ggpubr,get_legend)
NAMESPACE:importFrom(ggpubr,ggarrange)
NAMESPACE:importFrom(ggpubr,ggtexttable)
NAMESPACE:importFrom(ggpubr,text_grob)
NAMESPACE:importFrom(ggpubr,ttheme)
man/PlotGroups.Rd:p <- ggpubr::ggarrange(plotlist = c(list(gr_1$tree), gr_2),
man/GroupStats.Rd:p <- ggpubr::ggarrange(plotlist = c(list(gr_1$tree), gr_2),

```
