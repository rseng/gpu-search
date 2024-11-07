# https://github.com/voutcn/megahit

```console
CHANGELOG.md:-   Remove GPU support
CHANGELOG.md:-   `--cpu-only` turn on by default, and `--use-gpu` option to enable GPU
src/parallel_hashmap/phmap_config.h:        (defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 9) ||                \
src/parallel_hashmap/phmap_config.h:        (defined(__GNUC__) && !defined(__clang__) && !defined(__CUDACC__))
src/parallel_hashmap/phmap_config.h:    #elif defined(__CUDACC__)
src/parallel_hashmap/phmap_config.h:        #if __CUDACC_VER__ >= 70000
src/parallel_hashmap/phmap_config.h:        #endif  // __CUDACC_VER__ >= 70000
src/parallel_hashmap/phmap_config.h:    #endif  // defined(__CUDACC__)
src/sorting/read_to_sdbg.h:  int64_t words_per_substr_;  // substrings to be sorted by GPU
src/sorting/kmer_counter.h:  int64_t words_per_substr_{};  // substrings to be sorted by GPU
src/megahit:                                    'gpu-mem=',
src/megahit:                                    'use-gpu'])
src/megahit:                        '--use-gpu', '--gpu-mem'):

```
