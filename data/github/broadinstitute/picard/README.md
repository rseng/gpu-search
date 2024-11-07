# https://github.com/broadinstitute/picard

```console
src/test/java/picard/sam/CramCompatibilityTest.java:                        "RGID=4 RGLB=lib1 RGPL=illumina RGPU=unit1 RGSM=20",
src/test/java/picard/sam/CramCompatibilityTest.java:                        "RGID=4 RGLB=lib1 RGPL=illumina RGPU=unit1 RGSM=20",
src/test/java/picard/sam/CramCompatibilityTest.java:                {"picard.sam.AddOrReplaceReadGroups", "RGID=4 RGLB=lib1 RGPL=illumina RGPU=unit1 RGSM=20", CRAM_UNMAPPED},
src/main/java/picard/sam/AddOrReplaceReadGroups.java: *       RGPU=unit1 \
src/main/java/picard/sam/AddOrReplaceReadGroups.java:            "      RGPU=unit1 \\\n" +
src/main/java/picard/sam/AddOrReplaceReadGroups.java:    public String RGPU;
src/main/java/picard/sam/AddOrReplaceReadGroups.java:        rg.setPlatformUnit(RGPU);
src/main/java/picard/sam/AddOrReplaceReadGroups.java:        checkTagValue("RGPU", RGPU).ifPresent(validationFailures::add);
src/main/java/picard/illumina/IlluminaBasecallsToFastq.java:     * @return AsyncClusterWriter that contains one or more ClusterWriters (amount depends on read structure), all using

```
