# https://github.com/Bgi-LUSH/GPMeta

```console
data/plot_script/figureS7/box-plot.r:library(ggpubr)
data/plot_script/figure2/host-h.r:library(ggpubr)
data/plot_script/figureS3/box-plot-recall.r:library(ggpubr)
data/plot_script/figureS3/box-plot-precision.r:library(ggpubr)
data/plot_script/figureS5/plot.r:library(ggpubr)
data/plot_script/figureS2/plot-box-new.r:library(ggpubr)
data/README.md:Repository to reproduce analyses from manuscript titled "GPMeta: A GPU-accelerated method for ultrarapid pathogen identification from metagenomic sequences"
README.md:A GPU-accelerated method for ultrarapid pathogen identification from metagenomic sequences.
README.md:GPMeta is a GPU-accelerated method for ultrarapid pathogen identification from metagenomic sequences. GPMeta can rapidly and accurately remove host contamination, isolate microbial reads, and identify potential disease-causing pathogens. GPMeta is much faster than existing CPU-based tools, being 5-40x faster than Kraken2 and Centrifuge and 25-68x faster than Bwa and Bowtie2 by using a GPU-computing technique.
README.md:[Xuebin Wang, Taifu Wang, et al, GPMeta: a GPU-accelerated method for ultrarapid pathogen identification from metagenomic sequences, Briefings in Bioinformatics, 2023;bbad092, https://doi.org/10.1093/bib/bbad092.](https://doi.org/10.1093/bib/bbad092)
README.md:GPMeta requires NVIDIA's GPUs，which the sum of the graphics memory of the graphics card needs to be greater than the database size.
README.md:GPMeta requires database files in a specific format for sequence alignment. These files can be generated from FASTA formmated DNA sequence files.To generate multiple pathogen databases, you need to split the FASTA file evenly according to the number of GPU graphics cards(the minimum granularity is chromosomes).Database index can take step interval.Users have to prepare a database file in FASTA format and convert it into GPMeta format database files by using "Index the reference database" command at first.The database needs to be loaded into GPU video memory before sequence alignment.
README.md:#### load the reference database to GPU
README.md:    --batchSize        -b INT batch size of GPU memory [10000]
README.md:    --gpuRes           -g INT GPU resource number [32]
README.md:    --batchSize        -b INT batch size of GPU memory [10000]
README.md:    --gpuRes           -g INT GPU resource number [32]

```
