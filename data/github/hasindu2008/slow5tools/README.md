# https://github.com/hasindu2008/slow5tools

```console
docs/workflows.md:buttery-eel -i extracted.blow5  -g /path/to/ont-guppy/bin/ --config dna_r9.4.1_450bps_sup.cfg --device 'cuda:all' -o extracted_sup.fastq #see https://github.com/Psy-Fer/buttery-eel/ for butter-eel options
docs/archive.md:Now base-call the original FAST5 files as well as the reconverted FAST5 files. Following are some example commands, but make sure to set the base-calling profile to match your dataset and the CPU/GPU device based on your system.
docs/archive.md:guppy_basecaller -c dna_r9.4.1_450bps_fast.cfg -i ${FAST5_DIR} -s fast5_basecalls/  -r --device cuda:all
docs/archive.md:guppy_basecaller -c dna_r9.4.1_450bps_fast.cfg -i s2f_fast5/ -s s2f_fast5_basecalls/ -r --device cuda:all
docs/archive.md:However, note that sometimes this test diff will cause false errors due to base-callers providing slightly different outputs in various circumstances (see https://github.com/hasindu2008/slow5tools/issues/70). We recently came through a situation where Guppy 4.4.1 on a system with multiple GPUs (GeForce 3090 and 3070) produced slightly different results, even on the same FAST5 input when run multiple times.
test/test_with_guppy.sh:$GUPPY_BASECALLER -c ${CONFIG} -i $FAST5_DIR -s $GUPPY_OUTPUT_ORIGINAL -r --device cuda:all || die "Guppy failed"
test/test_with_guppy.sh:$GUPPY_BASECALLER -c ${CONFIG}  -i $S2F_OUTPUT_DIR -s $GUPPY_OUTPUT_S2F -r --device cuda:all || die "Guppy failed"
.travis.yml:    #       secure: KT3XFGJDQaN4EIbjRqAxnNFcG3fpkSFKct32bupVdlTMbzoI2c9oPyYKmUESyJODZML2BAwKV1knXrzhU/8o+vEDmTFdQ5fRNNHP8kcx6RVIZMSsJ+rF2DDl7sitr0rndyPbT97ACbsSZSdRiWK8MKo6YeMUJNkhuBOjXISFl4uiNezK0HeFCepQdWwB8W7De/1kgNQa9ZL6O7deiB+6DDZOVF/jr/YvIxCjytGwhFW6E5/EDHVmTt+9aXRXVgffzq9Ltt5oS30uYfWNJfAsk+81XEnuZ7GUGSyN4N77Xk2cI1LD3E8m1kMHksd34tn0A97nT4CPPI+WoUootzLDh29fUmSDPXbqwCGFDk8FTVUSAuwxZQQAESJrFv9fVl1vWTh3Bj8NkzXAHgmTtaBCxeIUhN+pBX3d7QQMYGpUHGGdIENGpfVi9vnnAYSsYs1foJtfuhw534ejju4laZ0bXVUE0w5AAKEw8H1xA+lS413ugKwAXEX6naFAD6NkZaX3ISSVeewIqtIW6UFCNqt/U++OY9vcwzFDbUq/wEjZtNRLzXzHzaObomRsOKqldje3FxH9p1QIPZXWgLBx8FhdFG6P+Nilhifq7Dcgx1dybKGTWOOurXrPKVgC9bLtXe/49LVMk5TirYtj2gXmlofi1SD52T9HpZVWH9haJ+Rnjso=

```
