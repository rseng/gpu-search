# https://github.com/merenlab/anvio

```console
anvio/tests/sandbox/workflows/metagenomics/config-megahit-no-qc-all-against-all.json:        "--use-gpu": "",
anvio/tests/sandbox/workflows/metagenomics/config-megahit-no-qc-all-against-all.json:        "--gpu-mem": "",
anvio/tests/sandbox/workflows/metagenomics/config-megahit.json:        "--use-gpu": "",
anvio/tests/sandbox/workflows/metagenomics/config-megahit.json:        "--gpu-mem": "",
anvio/workflows/metagenomics/__init__.py:                                                  "--use-gpu", "--gpu-mem", "--keep-tmp-files",
anvio/workflows/metagenomics/Snakefile:            use_gpu = M.get_rule_param("megahit", "--use-gpu"),
anvio/workflows/metagenomics/Snakefile:            gpu_mem = M.get_rule_param("megahit", "--gpu-mem"),
anvio/workflows/metagenomics/Snakefile:                "{params.mem_flag} {params.use_gpu} {params.gpu_mem} {params.keep_tmp_files} " + \

```
