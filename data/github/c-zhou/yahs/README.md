# https://github.com/c-zhou/yahs

```console
README.md:    (java -jar -Xmx32G juicer_tools.1.9.9_jcuda.0.8.jar pre alignments_sorted.txt out.hic.part scaffolds_final.chrom.sizes) && (mv out.hic.part out.hic)
README.md:    (java -jar -Xmx32G juicer_tools.1.9.9_jcuda.0.8.jar pre out_JBAT.txt out_JBAT.hic.part <(cat out_JBAT.log  | grep PRE_C_SIZE | awk '{print $2" "$3}')) && (mv out_JBAT.hic.part out_JBAT.hic)
scripts/run_yahs.sh:juicer_tools="java -Xmx32G -jar /bin/juicer_tools.1.9.9_jcuda.0.8.jar pre"

```
