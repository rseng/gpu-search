# https://github.com/biod/sambamba

```console
doc/design.org:the latest LLVM support for just in time (JIT) compilation and GPUs.
BioD/bio/std/hts/sam/reader.d:    if (t[1]._seqprocmode) {
BioD/bio/std/hts/sam/reader.d:        _seqprocmode = true;
BioD/bio/std/hts/sam/reader.d:    bool _seqprocmode;
BioD/bio/std/hts/bam/reader.d:        _seqprocmode = true;
BioD/bio/std/hts/bam/reader.d:    package bool _seqprocmode; // available for bio.std.hts.bam.readrange;
BioD/bio/std/hts/bam/readrange.d:        if (_reader !is null && _reader._seqprocmode) {
BioD/bio/std/hts/bam/randomaccessmanager.d:        bool old_mode = _reader._seqprocmode;
BioD/bio/std/hts/bam/randomaccessmanager.d:        _reader._seqprocmode = true;
BioD/bio/std/hts/bam/randomaccessmanager.d:        _reader._seqprocmode = old_mode;

```
