package org.flag4jv3.ndarrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.random.RandomGenerator;

import static org.flag4jv3.ndarrays.Slice.*;
import static org.junit.jupiter.api.Assertions.*;

class LayoutSliceTests {

    // ------------------------------------------------------------------ oracle


    /// Enumerates every buffer position reachable under `layout`, in row-major index order.
    private static List<Integer> positions(Layout layout) {
        int rank = layout.rank();
        int[] dims = layout.shape().dims();
        int[] strides = layout.strides();

        for (int d : dims) if (d == 0) return List.of();

        List<Integer> out = new ArrayList<>();
        int[] idx = new int[rank];

        outer:
        while (true) {
            int p = layout.offset();
            for (int a = 0; a < rank; a++) p += idx[a]*strides[a];
            out.add(p);

            for (int a = rank - 1; a >= 0; a--) {
                if (++idx[a] < dims[a]) continue outer;
                idx[a] = 0;
            }
            break;
        }

        return out;
    }


    /// Independently computes the positions an index expression should select, by resolving each
    /// entry against the source shape and walking the resulting per-axis position lists directly.
    /// Deliberately shares no code with Layout#slice.
    private static List<Integer> expected(Layout src, Slice... indexExpr) {
        int rank = src.rank();
        int[] dims = src.shape().dims();
        int[] strides = src.strides();

        // Expand ELLIPSIS and trailing implicit ALL into a flat, ellipsis-free expression.
        int consuming = 0;
        boolean hasEllipsis = false;
        for (Slice ix : indexExpr) {
            if (ix == ELLIPSIS) {
                hasEllipsis = true;
            } else if (ix != NEW_AXIS) consuming++;
        }
        List<Slice> flat = new ArrayList<>();
        for (Slice ix : indexExpr) {
            if (ix == ELLIPSIS) {
                for (int k = 0; k < rank - consuming; k++) flat.add(ALL);
            } else {
                flat.add(ix);
            }
        }
        if (!hasEllipsis) for (int k = 0; k < rank - consuming; k++) flat.add(ALL);

        // Per-output-axis lists of buffer deltas; points fold straight into the base.
        int base = src.offset();
        List<int[]> axisDeltas = new ArrayList<>();
        int ax = 0;

        for (Slice ix : flat) {
            switch (ix) {
                case Point p -> {
                    int i = p.i() < 0 ? p.i() + dims[ax] : p.i();
                    base += i*strides[ax];
                    ax++;
                }
                case Range r -> {
                    List<Integer> sel = new ArrayList<>();
                    int n = dims[ax];
                    int step = r.step();
                    int start = r.start() == null ? (step > 0 ? 0 : n - 1)
                            : (r.start() < 0 ? r.start() + n : r.start());
                    int stop = r.stop() == null ? (step > 0 ? n : -1)
                            : (r.stop() < 0 ? r.stop() + n : r.stop());
                    stop = step > 0 ? Math.min(stop, n) : Math.max(stop, -1);
                    for (int i = start; step > 0 ? i < stop : i > stop; i += step)
                        sel.add(i*strides[ax]);
                    axisDeltas.add(sel.stream().mapToInt(Integer::intValue).toArray());
                    ax++;
                }
                case Marker.ALL -> {
                    int[] sel = new int[dims[ax]];
                    for (int i = 0; i < dims[ax]; i++) sel[i] = i*strides[ax];
                    axisDeltas.add(sel);
                    ax++;
                }
                case Marker.NEW_AXIS -> axisDeltas.add(new int[]{0});
                default -> throw new AssertionError("unexpected entry: " + ix);
            }
        }

        // Cartesian product in row-major order.
        List<Integer> out = new ArrayList<>(List.of(base));
        for (int[] deltas : axisDeltas) {
            List<Integer> next = new ArrayList<>(out.size()*deltas.length);
            for (int p : out) for (int d : deltas) next.add(p + d);
            out = next;
        }
        for (int[] deltas : axisDeltas) if (deltas.length == 0) return List.of();

        return out;
    }


    private static void assertSliceMatchesOracle(Layout src, Slice... indexExpr) {
        Layout view = src.slice(indexExpr);
        assertEquals(expected(src, indexExpr), positions(view),
                () -> "positions mismatch for " + Arrays.toString(indexExpr)
                        + " on " + src.shape() + " -> " + view.shape());
    }

    // ------------------------------------------------------------------ shapes

    @Nested
    @DisplayName("resulting shape")
    class Shapes {

        private final Layout a = Layout.contiguous(new Shape(4, 5, 6), 1);


        @Test
        void pointDropsAxis() {
            assertEquals(new Shape(5, 6), a.slice(point(1)).shape());
            assertEquals(new Shape(4, 6), a.slice(ALL, point(0)).shape());
            assertEquals(new Shape(4, 5), a.slice(ALL, ALL, point(0)).shape());
        }


        @Test
        void rangeKeepsAxis() {
            assertEquals(new Shape(2, 5, 6), a.slice(range(1, 3)).shape());
            assertEquals(new Shape(4, 3, 6), a.slice(ALL, range(0, 5, 2)).shape());
            assertEquals(new Shape(2, 5, 6), a.slice(from(2)).shape());
            assertEquals(new Shape(2, 5, 6), a.slice(to(2)).shape());
            assertEquals(new Shape(4, 5, 6), a.slice(rev()).shape());
        }


        @Test
        void newAxisInsertsExtentOne() {
            assertEquals(new Shape(1, 4, 5, 6), a.slice(NEW_AXIS).shape());
            assertEquals(new Shape(4, 1, 5, 6), a.slice(ALL, NEW_AXIS).shape());
            assertEquals(new Shape(1, 5, 6), a.slice(point(0), NEW_AXIS).shape());
            assertEquals(new Shape(1, 1, 4, 5, 6), a.slice(NEW_AXIS, NEW_AXIS).shape());
        }


        @Test
        void ellipsisFillsUncoveredAxes() {
            assertEquals(new Shape(4, 5), a.slice(ELLIPSIS, point(0)).shape());
            assertEquals(new Shape(5, 6), a.slice(point(0), ELLIPSIS).shape());
            assertEquals(new Shape(5), a.slice(point(0), ELLIPSIS, point(0)).shape());
            // Ellipsis may expand to nothing.
            assertEquals(new Shape(6), a.slice(point(0), point(0), ELLIPSIS).shape());
        }


        @Test
        void trailingAxesAreImplicitlyAll() {
            assertEquals(a.shape(), a.slice().shape());
            assertEquals(new Shape(5, 6), a.slice(point(1)).shape());
            assertEquals(new Shape(2, 6), a.slice(range(1, 3), point(0)).shape());
        }


        @Test
        void emptyRangesAreLegal() {
            assertEquals(new Shape(0, 5, 6), a.slice(range(2, 2)).shape());
            assertEquals(new Shape(0, 5, 6), a.slice(range(3, 1)).shape());       // wrong direction
            assertEquals(new Shape(4, 0, 6), a.slice(ALL, range(1, 3, -1)).shape());
        }


        @Test
        void allPointsYieldScalar() {
            Layout s = a.slice(point(0), point(0), point(0));
            assertEquals(0, s.rank());
            assertEquals(1, s.shape().numelIntValueExact());
        }
    }

    // ------------------------------------------------------------------ offsets and strides

    @Nested
    @DisplayName("offset and strides")
    class OffsetsAndStrides {

        private final Layout a = Layout.contiguous(new Shape(4, 5, 6), 1);  // strides (30, 6, 1)


        @Test
        void pointFoldsIntoOffset() {
            assertEquals(30, a.slice(point(1)).offset());
            assertEquals(60 + 12, a.slice(point(2), point(2)).offset());
            assertArrayEquals(new int[]{6, 1}, a.slice(point(1)).strides());
        }


        @Test
        void stepMultipliesStride() {
            assertArrayEquals(new int[]{60, 6, 1}, a.slice(range(0, 4, 2)).strides());
            assertArrayEquals(new int[]{30, 12, 1}, a.slice(ALL, range(0, 5, 2)).strides());
        }


        @Test
        void negativeStepGivesNegativeStrideAndStartsAtEnd() {
            Layout r = a.slice(rev());
            assertArrayEquals(new int[]{-30, 6, 1}, r.strides());
            assertEquals(90, r.offset());                    // starts at index 3 of axis 0
            assertEquals(new Shape(4, 5, 6), r.shape());
        }


        @Test
        void negativeIndicesCountFromEnd() {
            assertEquals(a.slice(point(3)).offset(), a.slice(point(-1)).offset());
            assertEquals(a.slice(range(2, 4)).offset(), a.slice(range(-2, 4)).offset());
            assertEquals(a.slice(range(0, 3)).shape(), a.slice(range(0, -1)).shape());
        }


        @Test
        void slicingIsComposable() {
            Layout once = a.slice(range(1, 3), point(2));
            Layout twice = a.slice(range(1, 3)).slice(ALL, point(2));
            assertEquals(once.shape(), twice.shape());
            assertEquals(once.offset(), twice.offset());
            assertArrayEquals(once.strides(), twice.strides());
        }


        @Test
        void emptyExprIsIdentity() {
            Layout s = a.slice();
            assertEquals(a.shape(), s.shape());
            assertEquals(a.offset(), s.offset());
            assertArrayEquals(a.strides(), s.strides());
        }


        @Test
        void itemSizeIsPreserved() {
            Layout c = Layout.contiguous(new Shape(4, 5), 2);
            assertEquals(2, c.slice(point(1), range(0, 3)).itemSize());
        }
    }

    // ------------------------------------------------------------------ classification

    @Nested
    @DisplayName("contiguity and writability of views")
    class Classification {

        private final Layout a = Layout.contiguous(new Shape(4, 5, 6), 1);


        @Test
        void newAxisPreservesContiguity() {
            assertTrue(a.slice(NEW_AXIS).isCContiguous());
            assertTrue(a.slice(ALL, NEW_AXIS).isCContiguous());
            assertTrue(a.slice(ELLIPSIS, NEW_AXIS).isCContiguous());
        }


        @Test
        void newAxisViewsRemainWritable() {
            assertTrue(a.slice(NEW_AXIS).isWritable());
            assertTrue(a.slice(NEW_AXIS, NEW_AXIS).isWritable());
        }


        @Test
        void leadingPointPreservesContiguity() {
            assertTrue(a.slice(point(1)).isCContiguous());
            assertTrue(a.slice(point(1), point(2)).isCContiguous());
        }


        @Test
        void stridedViewIsNotContiguous() {
            assertFalse(a.slice(ALL, ALL, range(0, 6, 2)).isContiguous());
            assertFalse(a.slice(rev()).isContiguous());
        }


        @Test
        void distinctViewsAreWritable() {
            assertTrue(a.slice(range(1, 3)).isWritable());
            assertTrue(a.slice(rev()).isWritable());
            assertTrue(a.slice(ALL, range(0, 5, 2)).isWritable());
        }
    }

    // ------------------------------------------------------------------ errors

    @Nested
    @DisplayName("rejected index expressions")
    class Errors {

        private final Layout a = Layout.contiguous(new Shape(4, 5, 6), 1);


        @Test
        void multipleEllipsesRejected() {
            assertThrows(IllegalArgumentException.class, () -> a.slice(ELLIPSIS, ELLIPSIS));
            assertThrows(IllegalArgumentException.class, () -> a.slice(ELLIPSIS, point(0), ELLIPSIS));
        }


        @Test
        void overConsumingExprRejected() {
            assertThrows(IllegalArgumentException.class, () -> a.slice(ALL, ALL, ALL, ALL));
            assertThrows(IllegalArgumentException.class,
                    () -> a.slice(point(0), point(0), point(0), point(0)));
        }


        @Test
        void newAxisDoesNotCountTowardConsumption() {
            assertDoesNotThrow(() -> a.slice(NEW_AXIS, ALL, ALL, ALL, NEW_AXIS));
        }


        @Test
        void outOfBoundsPointRejected() {
            assertThrows(IndexOutOfBoundsException.class, () -> a.slice(point(4)));
            assertThrows(IndexOutOfBoundsException.class, () -> a.slice(point(-5)));
            assertThrows(IndexOutOfBoundsException.class, () -> a.slice(ALL, point(5)));
        }


        @Test
        void outOfBoundsRangeStartRejected() {
            assertThrows(IndexOutOfBoundsException.class, () -> a.slice(range(4, 4)));
            assertThrows(IndexOutOfBoundsException.class, () -> a.slice(range(-5, 2)));
        }


        @Test
        void overshootingRangeStopIsClamped() {
            assertEquals(new Shape(2, 5, 6), a.slice(range(2, 100)).shape());
            assertEquals(new Shape(4, 5, 6), a.slice(range(0, Integer.MAX_VALUE)).shape());
        }


        @Test
        void zeroStepRejectedAtConstruction() {
            assertThrows(IllegalArgumentException.class, () -> range(0, 4, 0));
        }


        @Test
        void nullsRejected() {
            assertThrows(NullPointerException.class, () -> a.slice((Slice[]) null));
            assertThrows(NullPointerException.class, () -> a.slice(ALL, null));
        }
    }

    // ------------------------------------------------------------------ property-based

    @Nested
    @DisplayName("property: view reaches exactly the selected buffer positions")
    class Properties {

        private static final RandomGenerator RNG = RandomGenerator.getDefault();


        private static Layout randomSource(RandomGenerator rng) {
            int rank = 1 + rng.nextInt(4);
            int[] dims = new int[rank];
            for (int i = 0; i < rank; i++) dims[i] = 1 + rng.nextInt(4);
            Layout base = Layout.contiguous(new Shape(dims), 1);
            // Half the time, start from a non-trivial view so offsets and negative strides compose.
            return rng.nextBoolean() ? base : base.slice(rev());
        }


        private static Slice randomEntry(RandomGenerator rng, int n) {
            return switch (rng.nextInt(6)) {
                case 0 -> point(rng.nextInt(n));
                case 1 -> point(-1 - rng.nextInt(n));
                case 2 -> ALL;
                case 3 -> NEW_AXIS;
                case 4 -> {
                    int start = rng.nextInt(n);
                    int stop = rng.nextInt(n + 2);           // may overshoot -> clamped
                    yield range(start, stop, 1 + rng.nextInt(3));
                }
                default -> {
                    int start = rng.nextInt(n);
                    int stop = rng.nextInt(n + 1) - 1;       // may reach -1
                    yield range(start, stop, -1 - rng.nextInt(2));
                }
            };
        }


        private static Slice[] randomExpr(RandomGenerator rng, Layout src) {
            List<Slice> expr = new ArrayList<>();
            int ax = 0;
            while (ax < src.rank() && rng.nextInt(5) > 0) {
                Slice ix = randomEntry(rng, src.getSize(ax));
                expr.add(ix);
                if (ix != NEW_AXIS) ax++;
            }
            if (rng.nextInt(4) == 0) expr.add(ELLIPSIS);
            return expr.toArray(Slice[]::new);
        }


        @Test
        @DisplayName("randomized expressions match the brute-force oracle")
        void randomizedAgainstOracle() {
            for (int trial = 0; trial < 20_000; trial++) {
                Layout src = randomSource(RNG);
                assertSliceMatchesOracle(src, randomExpr(RNG, src));
            }
        }


        @Test
        @DisplayName("every reachable position lies within the source's reachable set")
        void viewNeverEscapesSource() {
            for (int trial = 0; trial < 5_000; trial++) {
                Layout src = randomSource(RNG);
                Set<Integer> valid = new HashSet<>(positions(src));
                Layout view = src.slice(randomExpr(RNG, src));
                for (int p : positions(view))
                    assertTrue(valid.contains(p),
                            () -> "view escaped source buffer at position " + p);
            }
        }


        @Test
        @DisplayName("composition: slicing twice equals slicing once")
        void compositionIsAssociative() {
            for (int trial = 0; trial < 5_000; trial++) {
                Layout src = randomSource(RNG);
                Slice[] first = randomExpr(RNG, src);
                Layout mid = src.slice(first);
                if (mid.rank() == 0 || mid.shape().numelIntValueExact() == 0) continue;

                Slice[] second = randomExpr(RNG, mid);
                assertSliceMatchesOracle(mid, second);
            }
        }


        @Test
        @DisplayName("writable views have no duplicate positions")
        void writableImpliesInjective() {
            for (int trial = 0; trial < 5_000; trial++) {
                Layout src = randomSource(RNG);
                Layout view = src.slice(randomExpr(RNG, src));
                if (!view.isWritable()) continue;

                List<Integer> ps = positions(view);
                assertEquals(ps.size(), new HashSet<>(ps).size(),
                        () -> "layout reported writable but aliases: " + view.shape()
                                + " strides " + Arrays.toString(view.strides()));
            }
        }
    }
}