# https://github.com/dynamicslab/pysindy

```console
examples/data/vonKarman_pod/pod_modes/makefile:DPROCMAP=0
examples/data/vonKarman_pod/pod_modes/makefile:ifeq ($(DPROCMAP),1)
examples/data/vonKarman_pod/pod_modes/makefile:	CORE := ${CORE} dprocmap.o
examples/data/vonKarman_pod/pod_modes/makefile: 	DUMMY:= $(shell cp $S/PARALLEL.dprocmap $S/PARALLEL)
examples/data/vonKarman_pod/pod_modes/makefile:$(OBJDIR)/dprocmap.o    :$S/dprocmap.f $S/DPROCMAP;     $(FC) -c $(FL2) $< -o $@

```
