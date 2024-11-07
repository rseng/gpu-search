# https://github.com/debian-tex/biber

```console
dist/linux_aarch64/build.sh:  --link=/usr/lib/aarch64-linux-gnu/libicudata.so \
dist/darwinlegacy_x86_64/build.sh:  --link=/opt/local/lib/libicudata.67.dylib \
dist/darwin_x86_64/build.sh:  --link=/opt/local/lib/libicudata.58.dylib \
dist/darwin_arm64/build.sh:  --link=/opt/homebrew/Cellar/icu4c/73.2/lib/libicudata.73.dylib \
testfiles/test.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
testfiles/test.bcf:      <bcf:field>origpublisher</bcf:field>
testfiles/test.bcf:      <bcf:field>origpublisher</bcf:field>
data/biber-tool.conf:      <field fieldtype="list" datatype="literal">origpublisher</field>
data/biber-tool.conf:      <field>origpublisher</field>
data/biber-tool.conf:      <field>origpublisher</field>
t/xdata.t:      \annotation{item}{publisher}{default}{1}{}{0}{bigpublisher}
t/related-entries.t:      \field{relatedtype}{origpubin}
t/tdata/set-legacy.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/set-legacy.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/set-legacy.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding3.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding3.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding3.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/general.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/general.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/general.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/dm-constraints.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/dm-constraints.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/options.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/options.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/options.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding6.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding6.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding6.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extradate.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/extradate.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extradate.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-bbl.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/full-bbl.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-bbl.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness6.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness6.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness6.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/translit.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/translit.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/translit.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-complex.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sort-complex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-complex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/remote-files.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/remote-files.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/remote-files.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/labelalphaname.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/labelalphaname.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/labelalphaname.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extratitleyear.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/extratitleyear.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extratitleyear.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/biber-test.conf:      <field fieldtype="list" datatype="literal">origpublisher</field>
t/tdata/biber-test.conf:      <field>origpublisher</field>
t/tdata/biber-test.conf:      <field>origpublisher</field>
t/tdata/xdata.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/xdata.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/xdata.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/set-dynamic.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/set-dynamic.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/set-dynamic.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding2.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding2.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding2.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/names.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/names.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/names.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-case.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sort-case.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-case.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-uc.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sort-uc.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-uc.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness3.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness3.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness3.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness-nameparts.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness-nameparts.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness-nameparts.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/skipsg.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/skipsg.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/skipsg.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/bibtex-aliases.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/bibtex-aliases.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/bibtex-aliases.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-names.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sort-names.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-names.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness2.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness2.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness2.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/dateformats.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/dateformats.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/dateformats.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/skips.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/skips.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/skips.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/names_x.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/names_x.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/names_x.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/related.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/related.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/related.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/crossrefs.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/crossrefs.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/crossrefs.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/truncation.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/truncation.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/truncation.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-bibtex.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/full-bibtex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-bibtex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding4.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding4.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding4.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/xdata.bib:  PUBLISHER+an = {1=bigpublisher}
t/tdata/encoding5.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding5.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding5.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness4.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness4.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness4.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sections-complex.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sections-complex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sections-complex.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness5.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness5.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness5.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/tool-test.conf:      <field fieldtype="list" datatype="literal">origpublisher</field>
t/tdata/tool-test.conf:      <field>origpublisher</field>
t/tdata/tool-test.conf:      <field>origpublisher</field>
t/tdata/maps.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/maps.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/maps.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/datalists.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/datalists.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/datalists.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sections.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sections.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sections.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-order.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/sort-order.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/sort-order.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extratitle.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/extratitle.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/extratitle.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/labelalpha.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/labelalpha.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/labelalpha.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/related.bib:  RELATEDTYPE  = {origpubin},
t/tdata/annotations.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/annotations.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/annotations.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/biblatexml.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/biblatexml.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/biblatexml.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/set-static.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/set-static.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/set-static.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding1.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/encoding1.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/encoding1.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/basic-misc.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/basic-misc.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/basic-misc.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-dot.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/full-dot.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/full-dot.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness7.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness7.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness7.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/bibtex-output.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/bibtex-output.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/bibtex-output.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness1.bcf:      <bcf:field fieldtype="list" datatype="literal">origpublisher</bcf:field>
t/tdata/uniqueness1.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/uniqueness1.bcf:      <bcf:field>origpublisher</bcf:field>
t/tdata/tool-testsort.conf:      <field fieldtype="list" datatype="literal">origpublisher</field>
t/tdata/tool-testsort.conf:      <field>origpublisher</field>
t/tdata/tool-testsort.conf:      <field>origpublisher</field>

```
