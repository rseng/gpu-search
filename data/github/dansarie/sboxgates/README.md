# https://github.com/dansarie/sboxgates

```console
sboxgates.c:    "S-box. Generated graphs can be converted to C/CUDA source code or to Graphviz DOT format.\v"
sboxgates.c:  {"convert-c",       'c',            0, 0, "Convert input file to a C or CUDA function.", 2},
convert_graph.c:   Helper functions for converting generated graphs to C/CUDA code or Graphviz dot format for
convert_graph.c:  /* Generate CUDA code if LUT gates are present. */
convert_graph.c:  bool cuda = false;
convert_graph.c:      cuda = true;
convert_graph.c:  if (cuda) {
convert_graph.c:  if (cuda) {
README.md:implementations for use on Nvidia GPUs that support the LOP3.LUT instruction, or on FPGAs.
README.md:The program can convert the XML files to C or CUDA functions. This is enabled by the `-c`
README.md:argument. Graphs that include at least one LUT are converted to CUDA functions and graphs without
README.md:Convert a generated circuit to C/CUDA:
convert_graph.h:/* Converts a gate network to a C or CUDA function and prints it to stdout. If the state contains
convert_graph.h:   at least one LUT gate it will be converted to a CUDA function. Otherwise, it will be converted to
.travis.yml:      - nvidia-cuda-toolkit

```
