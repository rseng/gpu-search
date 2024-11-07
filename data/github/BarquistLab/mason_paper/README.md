# https://github.com/BarquistLab/mason_paper

```console
data/reference_sequences/FQ312003.1.gff:FQ312003.1	EMBL	gene	2857000	2857485	.	-	.	ID=gene-SL1344_2667;Name=gpU;gbkey=Gene;gene=gpU;gene_biotype=protein_coding;locus_tag=SL1344_2667
data/reference_sequences/FQ312003.1.gff:FQ312003.1	EMBL	CDS	2857000	2857485	.	-	0	ID=cds-CBW18769.1;Parent=gene-SL1344_2667;Dbxref=EnsemblGenomes-Gn:SL1344_2667,EnsemblGenomes-Tr:CBW18769,InterPro:IPR009734,InterPro:IPR016912,UniProtKB/TrEMBL:E1WJI1,NCBI_GP:CBW18769.1;Name=CBW18769.1;gbkey=CDS;gene=gpU;inference=similar to AA sequence:INSD:ADX18469.1;locus_tag=SL1344_2667;product=hypothetical bacteriophage tail protein;protein_id=CBW18769.1;transl_table=11
data/salm_offtargets_w_tm.csv:"3876","SL1344_2667","gpU","-",-13,"GGGGTATGAA","JVpna-15_acpP","AGAGTATGAG",3,"1;3;10",6,"GTATGA",1.92337172675419,-0.70078688377133,0.523199073475053,NA,"no",-8.2,6.74318505391011
data/salm_offtargets_w_tm.csv:"10853","SL1344_2667","gpU","-",-3,"TCATGATGAT","Jvpna-37_acpP_PNA_mut_9","TCAGTATGAG",3,"4;5;10",4,"ATGA",1.92337172675419,-0.270507791259853,0.256358938112684,NA,"no",-8.2,-50.4185736708241
data/nonessentialgenes_start.gff:FQ312003.1	EMBL	CDS	2857475	2857492	.	-	0	ID=cds-CBW18769.1;Parent=gene-SL1344_2667;Dbxref=EnsemblGenomes-Gn:SL1344_2667,EnsemblGenomes-Tr:CBW18769,InterPro:IPR009734,InterPro:IPR016912,UniProtKB/TrEMBL:E1WJI1,NCBI_GP:CBW18769.1;Name=CBW18769.1;gbkey=CDS;gene=gpU;inference=similar to AA sequence:INSD:ADX18469.1;locus_tag=SL1344_2667;product=hypothetical bacteriophage tail protein;protein_id=CBW18769.1;transl_table=11
data/link_lt_gn.tab:gpU	SL1344_2667
data/mismatches_02_2021/pna_3mm_startregions.tab:SL1344_2667	19	GGGGTATGAA	JVpna-15_acpP	AGAGTATGAG	3	gpU	1;3;10
data/mismatches_02_2021/pna_3mm_startregions.tab:SL1344_2667	29	TCATGATGAT	Jvpna-37_acpP_PNA_mut_9	TCAGTATGAG	3	gpU	4;5;10
data/mismatches_02_2021/nonessentialgenes.gff:FQ312003.1	EMBL	CDS	2857465	2857502	.	-	0	ID=cds-CBW18769.1;Parent=gene-SL1344_2667;Dbxref=EnsemblGenomes-Gn:SL1344_2667,EnsemblGenomes-Tr:CBW18769,InterPro:IPR009734,InterPro:IPR016912,UniProtKB/TrEMBL:E1WJI1,NCBI_GP:CBW18769.1;Name=CBW18769.1;gbkey=CDS;gene=gpU;inference=similar to AA sequence:INSD:ADX18469.1;locus_tag=SL1344_2667;product=hypothetical bacteriophage tail protein;protein_id=CBW18769.1;transl_table=11
data/mismatches_02_2021/all_genes_startregions.gff:FQ312003.1	EMBL	CDS	2857465	2857502	.	-	0	ID=cds-CBW18769.1;Parent=gene-SL1344_2667;Dbxref=EnsemblGenomes-Gn:SL1344_2667,EnsemblGenomes-Tr:CBW18769,InterPro:IPR009734,InterPro:IPR016912,UniProtKB/TrEMBL:E1WJI1,NCBI_GP:CBW18769.1;Name=CBW18769.1;gbkey=CDS;gene=gpU;inference=similar to AA sequence:INSD:ADX18469.1;locus_tag=SL1344_2667;product=hypothetical bacteriophage tail protein;protein_id=CBW18769.1;transl_table=11
data/mismatches_02_2021/start_regions.gff:FQ312003.1	EMBL	CDS	2857470	2857515	.	-	0	ID=cds-CBW18769.1;Parent=gene-SL1344_2667;Dbxref=EnsemblGenomes-Gn:SL1344_2667,EnsemblGenomes-Tr:CBW18769,InterPro:IPR009734,InterPro:IPR016912,UniProtKB/TrEMBL:E1WJI1,NCBI_GP:CBW18769.1;Name=CBW18769.1;gbkey=CDS;gene=gpU;inference=similar to AA sequence:INSD:ADX18469.1;locus_tag=SL1344_2667;product=hypothetical bacteriophage tail protein;protein_id=CBW18769.1;transl_table=11
data/mismatches_02_2021/link_lt_gn.tab:gpU	SL1344_2667
analysis/diff_exp_rawdata/acpP_PNA_mut_4_vs_ctrl.csv:"gpU",-0.821083708457208,1.92337172675419,5.77891211013456,0.0220684457745926,0.28342692724869,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_6_vs_ctrl.csv:"gpU",-0.169131728924194,1.92337172675419,0.271333898683712,0.605958471668551,0.80191106047888,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_7_vs_ctrl.csv:"gpU",-0.326090319072827,1.92337172675419,0.900505139600225,0.349627759153315,0.553921407136272,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_8_vs_ctrl.csv:"gpU",-0.776345240148982,1.92337172675419,5.1560999977733,0.0299021974079676,0.109891713305232,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_9_vs_ctrl.csv:"gpU",-0.270507791259853,1.92337172675419,0.742269247569003,0.395227358045823,0.554167512196074,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_3_vs_ctrl.csv:"gpU",-0.338319265112045,1.92337172675419,1.11190738716745,0.299419666543186,0.573856833986198,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/KFF_ompC_vs_ctrl.csv:"gpU",0.036092225140036,2.34852483621539,0.0312988346315368,0.861144534336189,0.887407905694456,"gpU"
analysis/diff_exp_rawdata/KFF_rpl32_vs_ctrl.csv:"gpU",-0.133770847809849,2.34852483621539,0.39205384678554,0.537468355824492,0.624067164222905,"gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_1_vs_ctrl.csv:"gpU",-0.0123885330873478,1.92337172675419,0.00138008508061031,0.970592342580524,0.985875652126171,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/KFF_ackA_vs_ctrl.csv:"gpU",0.153731082317711,2.34852483621539,0.572699741813695,0.456972274758309,0.526973042220389,"gpU"
analysis/diff_exp_rawdata/acpP_scrambled_vs_ctrl.csv:"gpU",-0.427161543893478,1.92337172675419,1.71456364319863,0.199547408438609,0.587211721724583,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/KFF_penta_vs_ctrl.csv:"gpU",-0.0831079104546038,2.34852483621539,0.148088328587242,0.703951252568041,0.755829569072501,"gpU"
analysis/diff_exp_rawdata/KFF_ddlB_vs_ctrl.csv:"gpU",-0.109635589943763,2.34852483621539,0.264328839369561,0.612131301678681,0.697662371870825,"gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_5_vs_ctrl.csv:"gpU",-0.939488473085087,1.92337172675419,7.0756266591243,0.0120216389985286,0.180848893220855,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_vs_ctrl.csv:"gpU",-0.70078688377133,1.92337172675419,4.40289117221262,0.0437096747417919,0.299778806706421,"SL1344_2667","gpU"
analysis/diff_exp_rawdata/acpP_PNA_mut_2_vs_ctrl.csv:"gpU",-0.179159721556562,1.92337172675419,0.337057596813615,0.565524190620576,0.665752393102575,"SL1344_2667","gpU"
analysis/ot_salm.csv:"529","SL1344_2667","gpU","-",-13,"GGGGTATGAA","JVpna-15_acpP","AGAGTATGAG",3,"1;3;10",6,"GTATGA",1.92337172675419,-0.70078688377133,0.523199073475053,NA,"no",2.5
scripts/mason_RNASEQ.Rmd:library(ggpubr)

```
