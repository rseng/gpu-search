# https://github.com/Tomcxf/FASTdRNA

```console
docs/_posts/2024-05-08-main.md:For f5c, it offers two version called f5c_x86_64_linux and f5c_x86_64_linux_cuda.
docs/_posts/2024-05-08-main.md:cuda version is for GPU, so if you only run with CPU, please change dRNAmain.py line 159
docs/_posts/2024-05-08-main.md:"f5c_x86_64_linux_cuda index --slow5 {input[1]} {input[0]}"
script/dRNAtail.py:        "f5c_x86_64_linux_cuda index --slow5 {input[1]} {input[0]}"
script/dRNAmain.py:        "cuda:0"
script/dRNAmain.py:#        "f5c_x86_64_linux_cuda index --slow5 {input[1]} {input[0]}"
script/dRNAmodif_1.py:        "f5c_x86_64_linux_cuda index --slow5 {input[1]} {input[0]}"
script/dRNAmodif_1.py:        "f5c_x86_64_linux_cuda eventalign  --print-read-names --scale-events --samples --slow5 {input[1]} -b {input[2]} -g {input[3]} -r {input[0]} -t 8 --rna > {output}"

```
