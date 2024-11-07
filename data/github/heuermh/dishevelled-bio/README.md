# https://github.com/heuermh/dishevelled-bio

```console
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    protected static final Range<Integer> openClosed = Range.openClosed(1, 100);
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testCountEmptyOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(0, create(empty).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testCountSingletonOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(0, create(singleton).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testCountSparseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(2, create(sparse).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testCountDenseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(101, create(dense).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testIntersectEmptyOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertTrue(Iterables.isEmpty(create(empty).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testIntersectSingletonOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(0, Iterables.size(create(singleton).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testIntersectSparseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(2, Iterables.size(create(sparse).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:    public void testIntersectDenseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/tree/AbstractRangeTreeTest.java:        assertEquals(101, Iterables.size(create(dense).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    protected static final Range<Integer> openClosed = Range.openClosed(1, 100);
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testCountEmptyOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(0, create(empty).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testCountSingletonOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(0, create(singleton).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testCountSparseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(2, create(sparse).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testCountDenseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(101, create(dense).count(openClosed));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testIntersectEmptyOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertTrue(Iterables.isEmpty(create(empty).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testIntersectSingletonOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(0, Iterables.size(create(singleton).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testIntersectSparseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(2, Iterables.size(create(sparse).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:    public void testIntersectDenseOpenClosedRange() {
range/src/test/java/org/dishevelled/bio/range/entrytree/AbstractRangeTreeTest.java:        assertEquals(101, Iterables.size(create(dense).intersect(openClosed)));
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:import static org.dishevelled.bio.range.rtree.RangeGeometries.openClosed;
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:    public void testOpenClosedNullLower() {
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:        openClosed(null, 42L);
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:    public void testOpenClosedNullUpper() {
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:        openClosed(24L, null);
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:    public void testOpenClosed() {
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:        assertEquals(Geometries.rectangle(25.0d, 0.0d, 42.0d, 1.0d), openClosed(24L, 42L));
range/src/test/java/org/dishevelled/bio/range/rtree/RangeGeometriesTest.java:        assertEquals(Geometries.rectangle(25.0d, 0.0d, 42.0d, 1.0d), range(Range.openClosed(24L, 42L)));
range/src/main/java/org/dishevelled/bio/range/rtree/RangeGeometries.java:    public static <N extends Number & Comparable<? super N>> Rectangle openClosed(final N lower, final N upper) {
range/src/main/java/org/dishevelled/bio/range/rtree/RangeGeometries.java:        return range(Range.openClosed(lower, upper));
range/src/main/java/org/dishevelled/bio/range/rtree/RangeGeometries.java:          openClosed(10, 20) --> (11.0, 0.0, 20.0, 1.0)
range/src/main/java/org/dishevelled/bio/range/rtree/RangeGeometries.java:          openClosed(10, 11) --> (11.0, 0.0, 11.0, 1.0)
range/src/main/java/org/dishevelled/bio/range/rtree/RangeGeometries.java:          openClosed(10, 10) --> empty, throw exception

```
