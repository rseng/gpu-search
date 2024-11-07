# https://github.com/vgteam/vg

```console
test/t/34_vg_pack.t:is $(cat pileup/2snp.gam.vgpu.json | jq '.node_pileups[].base_pileup[] | (.num_bases // 0)' | awk '{ print NR-1, $0 }' | head | md5sum | cut -f 1 -d\ )\
test/t/34_vg_pack.t:rm -f flat.vg 2snp.vg 2snp.xg 2snp.sim flat.gcsa flat.gcsa.lcp flat.xg 2snp.xg 2snp.gam 2snp.gam.cx 2snp.gam.cx.3x 2snp.gam.vgpu
test/t/34_vg_pack.t:x=$(cat pileup/tiny.vgpu.json | jq  '.edge_pileups' | grep num_reads | awk '{print $2}' | sed -e 's/\,//' | awk '{sum+=$1} END {print sum}')
test/t/34_vg_pack.t:rm -f tiny.vg tiny.xg tiny.gam tiny.vgpu tiny.pack
.gitignore:/*.vgpu
src/unittest/catch.hpp:#if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC) && !defined(__CUDACC__) && !defined(__LCC__)
src/unittest/catch.hpp:#  if !defined(__ibmxl__) && !defined(__CUDACC__)
src/unittest/catch.hpp:        // Code accompanying the article "Approximating the erfinv function" in GPU Computing Gems, Volume 2

```
