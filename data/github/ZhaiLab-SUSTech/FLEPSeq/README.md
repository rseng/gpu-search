# https://github.com/ZhaiLab-SUSTech/FLEPSeq

```console
README.md:You can use MinKNOW to perform real-time basecalling while sequencing, or use the GPU version of Guppy to speed up basecalling after sequencing. Both MinKNOW and Guppy are available via Nanopore community site (https://community.nanoporetech.com). Command-line for running Guppy basecalling is as follow:
README.md:$ guppy_basecaller -i raw_fast5_dir -s out_fast5_dir -c dna_r9.4.1_450bps_hac.cfg --recursive --fast5_out --disable_pings --qscore_filtering --device "cuda:all:100%"

```
