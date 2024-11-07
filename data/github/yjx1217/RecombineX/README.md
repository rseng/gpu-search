# https://github.com/yjx1217/RecombineX

```console
pipelines/RecombineX.02.Polymorphic_Markers_by_Reference_based_Read_Mapping.sh:    -RGPU "$parent1_tag" \
pipelines/RecombineX.02.Polymorphic_Markers_by_Reference_based_Read_Mapping.sh:    -RGPU "$parent2_tag" \
pipelines/RecombineX.13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping.sh:    -RGPU "$parent1_based_prefix" \
pipelines/RecombineX.13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping.sh:    -RGPU "$parent2_based_prefix" \
scripts/batch_read_mapping_to_reference_genome.pl:	system("$java_dir/java -Djava.io.tmpdir=./tmp -Dpicard.useLegacyParser=false -XX:ParallelGCThreads=$threads -jar $picard_dir/picard.jar AddOrReplaceReadGroups -INPUT=${sample_tag}.${reference_genome_tag}.fixmate.bam -OUTPUT ${sample_tag}.${reference_genome_tag}.rdgrp.bam -SORT_ORDER coordinate -MAX_RECORDS_IN_RAM 1000000 -RGID ${sample_tag}.${reference_genome_tag} -RGLB ${sample_tag}.${reference_genome_tag} -RGPL 'Illumina' -RGPU ${sample_tag}.${reference_genome_tag} -RGSM ${sample_tag} -RGCN 'RGCN'");
scripts/batch_read_mapping_to_parent_genomes.pl:	system("$java_dir/java -Djava.io.tmpdir=./tmp -Dpicard.useLegacyParser=false -XX:ParallelGCThreads=$threads -jar $picard_dir/picard.jar AddOrReplaceReadGroups -INPUT ${sample_tag}.${parent_genome_tag}.fixmate.bam -OUTPUT ${sample_tag}.${parent_genome_tag}.rdgrp.bam -SORT_ORDER coordinate -MAX_RECORDS_IN_RAM 1000000 -RGID ${sample_tag}.${parent_genome_tag} -RGLB ${sample_tag}.${parent_genome_tag} -RGPL 'Illumina' -RGPU ${sample_tag}.${parent_genome_tag} -RGSM ${sample_tag} -RGCN 'RGCN'");
Project_Template/02.Polymorphic_Markers_by_Reference_based_Read_Mapping/RecombineX.02.Polymorphic_Markers_by_Reference_based_Read_Mapping.sh:    -RGPU "$parent1_tag" \
Project_Template/02.Polymorphic_Markers_by_Reference_based_Read_Mapping/RecombineX.02.Polymorphic_Markers_by_Reference_based_Read_Mapping.sh:    -RGPU "$parent2_tag" \
Project_Template/13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping/RecombineX.13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping.sh:    -RGPU "$parent1_based_prefix" \
Project_Template/13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping/RecombineX.13.Polymorphic_Markers_by_Cross_Parent_Read_Mapping.sh:    -RGPU "$parent2_based_prefix" \

```
