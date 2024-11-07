# https://github.com/zhouzilu/DENDRO

```console
script/mutation_detection_mapreduce.sh:java -jar pathtopicard/picard/picard.jar AddOrReplaceReadGroups INPUT=${SAMPLE}.sorted.bam OUTPUT=${SAMPLE}.sorted.rg.bam RGID=${SAMPLE} RGLB=trancriptome RGPL=ILLUMINA RGPU=machine RGSM=${SAMPLE}

```
