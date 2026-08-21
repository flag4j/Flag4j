package org.flag4jv3.ndarrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for [Layout].
///
/// Several tests use a brute-force oracle: enumerate every logical nD slice, map it through
/// [Layout#toBufferIndex(int, int...)], and check a property of the resulting slot set. This is
/// the strongest available check on the layout invariants (injectivity, extent bounds, and
/// mapping preservation under [Layout#squeeze()]) because it does not reimplement the stride
/// arithmetic being tested, it only enumerates the slice space.
@DisplayName("Layout")
class LayoutTests {

    // =====================================================================================
    // Construction & validation
    // =====================================================================================
    @Nested
    @DisplayName("construction")
    class Construction {

        @Test
        @DisplayName("accessors reflect the constructor arguments")
        void basicAccessors() {
            Shape shape = new Shape(2, 3, 4);
            Layout layout = new Layout(shape, 5, new int[]{12, 4, 1}, 2);

            assertEquals(shape, layout.shape());
            assertEquals(5, layout.offset());
            assertArrayEquals(new int[]{12, 4, 1}, layout.strides());
            assertEquals(2, layout.itemSize());
            assertEquals(3, layout.rank());
            assertEquals(3, layout.getSize(1));
            assertEquals(4, layout.stride(1));
        }


        @Test
        @DisplayName("null shape or strides are rejected")
        void nullsRejected() {
            assertThrows(NullPointerException.class,
                    () -> new Layout(null, 0, new int[]{1}, 1));
            assertThrows(NullPointerException.class,
                    () -> new Layout(new Shape(3), 0, null, 1));
        }


        @Test
        @DisplayName("negative offset is rejected")
        void negativeOffsetRejected() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Layout(new Shape(3), -1, new int[]{1}, 1));
        }


        @ParameterizedTest
        @ValueSource(ints = {0, -1, -8})
        @DisplayName("non-positive itemSize is rejected")
        void badItemSizeRejected(int itemSize) {
            assertThrows(IllegalArgumentException.class,
                    () -> new Layout(new Shape(3), 0, new int[]{1}, itemSize));
        }


        @Test
        @DisplayName("stride count must match the shape's rank")
        void strideRankMismatchRejected() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Layout(new Shape(2, 3), 0, new int[]{1}, 1));
            assertThrows(IllegalArgumentException.class,
                    () -> new Layout(new Shape(2, 3), 0, new int[]{6, 3, 1}, 1));
        }


        @Test
        @DisplayName("layout does not alias the caller's strides array")
        void stridesDefensivelyCopied() {
            int[] strides = {3, 1};
            Layout layout = new Layout(new Shape(2, 3), 0, strides, 1);
            strides[0] = 99;

            assertArrayEquals(new int[]{3, 1}, layout.strides());
            assertEquals(3, layout.stride(0));
        }


        @Test
        @DisplayName("strides() returns a copy the caller cannot use to mutate the layout")
        void stridesAccessorReturnsCopy() {
            Layout layout = Layout.contiguous(new Shape(2, 3), 1);
            layout.strides()[0] = 99;

            assertArrayEquals(new int[]{3, 1}, layout.strides());
        }
    }


    // =====================================================================================
    // Contiguous factories
    // =====================================================================================
    @Nested
    @DisplayName("contiguous factories")
    class ContiguousFactories {

        @Test
        @DisplayName("default factory produces a C-contiguous, zero-offset layout")
        void defaultsToC() {
            Layout layout = Layout.contiguous(new Shape(2, 3, 4), 1);

            assertEquals(0, layout.offset());
            assertArrayEquals(new int[]{12, 4, 1}, layout.strides());
            assertTrue(layout.isCContiguous());
            assertEquals(ContiguousOrder.C, layout.contiguousOrder());
        }


        @Test
        @DisplayName("F factory produces column-major strides and reports F order")
        void fOrderFactory() {
            Layout layout = Layout.contiguous(new Shape(2, 3, 4), 1, ContiguousOrder.F);

            assertArrayEquals(new int[]{1, 2, 6}, layout.strides());
            assertTrue(layout.isFContiguous());
            assertFalse(layout.isCContiguous());
            assertEquals(ContiguousOrder.F, layout.contiguousOrder());
        }


        @ParameterizedTest
        @ValueSource(strings = {"BOTH", "NONE"})
        @DisplayName("factory rejects orders that are not exactly C or F")
        void rejectsNonCorF(String orderName) {
            ContiguousOrder order = ContiguousOrder.valueOf(orderName);

            assertThrows(IllegalArgumentException.class,
                    () -> Layout.contiguous(new Shape(2, 3), 1, order));
        }


        @Test
        @DisplayName("itemSize does not affect strides (strides are in element units)")
        void itemSizeDoesNotAffectStrides() {
            Layout real = Layout.contiguous(new Shape(2, 3), 1);
            Layout complex = Layout.contiguous(new Shape(2, 3), 2);
            Layout quaternion = Layout.contiguous(new Shape(2, 3), 4);

            assertArrayEquals(real.strides(), complex.strides());
            assertArrayEquals(real.strides(), quaternion.strides());
        }


        @Test
        @DisplayName("contiguous layouts are writable")
        void contiguousIsWritable() {
            assertTrue(Layout.contiguous(new Shape(2, 3, 4), 1).isWritable());
            assertTrue(Layout.contiguous(new Shape(2, 3, 4), 2, ContiguousOrder.F).isWritable());
        }


        @Test
        @DisplayName("asContiguous preserves shape and itemSize and drops offset/strides")
        void asContiguous() {
            Layout strided = new Layout(new Shape(2, 3), 7, new int[]{10, 2}, 2);
            Layout contiguous = strided.asContiguous();

            assertEquals(strided.shape(), contiguous.shape());
            assertEquals(strided.itemSize(), contiguous.itemSize());
            assertEquals(0, contiguous.offset());
            assertArrayEquals(new int[]{3, 1}, contiguous.strides());
        }
    }


    // =====================================================================================
    // Contiguity classification
    // =====================================================================================
    @Nested
    @DisplayName("contiguity classification")
    class Contiguity {

        @Test
        @DisplayName("canonical C strides classify as C only")
        void cOnly() {
            Layout layout = new Layout(new Shape(2, 3, 4), 0, new int[]{12, 4, 1}, 1);

            assertEquals(ContiguousOrder.C, layout.contiguousOrder());
            assertTrue(layout.isContiguous());
            assertTrue(layout.isContiguous(ContiguousOrder.C));
            assertFalse(layout.isContiguous(ContiguousOrder.F));
        }


        @Test
        @DisplayName("canonical F strides classify as F only")
        void fOnly() {
            Layout layout = new Layout(new Shape(2, 3, 4), 0, new int[]{1, 2, 6}, 1);

            assertEquals(ContiguousOrder.F, layout.contiguousOrder());
            assertTrue(layout.isFContiguous());
            assertFalse(layout.isCContiguous());
        }


        @ParameterizedTest(name = "shape {0}")
        @MethodSource("bothOrderShapes")
        @DisplayName("rank<=1 and all-extent-1 layouts are both C- and F-contiguous")
        void bothOrders(int[] dims) {
            Shape shape = new Shape(dims);
            Layout layout = Layout.contiguous(shape, 1);

            assertEquals(ContiguousOrder.BOTH, layout.contiguousOrder());
            assertTrue(layout.isCContiguous());
            assertTrue(layout.isFContiguous());
        }


        static Stream<Arguments> bothOrderShapes() {
            return Stream.of(
                    Arguments.of((Object) new int[0]),         // scalar
                    Arguments.of((Object) new int[]{5}),       // vector
                    Arguments.of((Object) new int[]{1, 1, 1}), // all singleton
                    Arguments.of((Object) new int[]{1, 5, 1})  // one real axis
            );
        }


        @Test
        @DisplayName("empty (zero-extent) layouts are both C- and F-contiguous")
        void emptyIsBoth() {
            Layout layout = new Layout(new Shape(2, 0, 3), 0, new int[]{7, 5, 3}, 1);

            assertEquals(ContiguousOrder.BOTH, layout.contiguousOrder());
        }


        @Test
        @DisplayName("extent-1 axes carry arbitrary strides without breaking contiguity")
        void extentOneAxesIgnored() {
            // Logical layout of a (2,1,3) C-contiguous array; the middle stride is arbitrary
            // because that axis is never stepped.
            Layout layout = new Layout(new Shape(2, 1, 3), 0, new int[]{3, 999, 1}, 1);

            assertTrue(layout.isCContiguous(),
                    "extent-1 axes must not be allowed to defeat the contiguity predicate");
        }


        @Test
        @DisplayName("gaps, negative strides, and permuted strides classify as NONE")
        void nonContiguous() {
            assertEquals(ContiguousOrder.NONE,
                    new Layout(new Shape(2, 3), 0, new int[]{6, 2}, 1).contiguousOrder(),
                    "gapped strides");
            assertEquals(ContiguousOrder.NONE,
                    new Layout(new Shape(2, 3), 5, new int[]{3, -1}, 1).contiguousOrder(),
                    "reversed inner axis");
            assertEquals(ContiguousOrder.NONE,
                    new Layout(new Shape(2, 3, 4), 0, new int[]{4, 8, 1}, 1).contiguousOrder(),
                    "permuted strides");
        }


        @Test
        @DisplayName("a non-zero offset does not affect contiguity classification")
        void offsetIrrelevantToContiguity() {
            Layout layout = new Layout(new Shape(2, 3), 17, new int[]{3, 1}, 1);

            assertTrue(layout.isCContiguous());
        }


        @Test
        @DisplayName("areContiguousAndMatchOrder requires both contiguous AND same order")
        void matchOrder() {
            Shape shape = new Shape(3, 4);
            Layout c1 = Layout.contiguous(shape, 1);
            Layout c2 = Layout.contiguous(shape, 1);
            Layout f = Layout.contiguous(shape, 1, ContiguousOrder.F);
            Layout none = new Layout(shape, 0, new int[]{8, 2}, 1);
            Layout vector = Layout.contiguous(new Shape(12), 1); // order BOTH

            assertTrue(Layout.areContiguousAndMatchOrder(c1, c2));
            assertFalse(Layout.areContiguousAndMatchOrder(c1, f),
                    "C and F must not be treated as a matching pair");
            assertFalse(Layout.areContiguousAndMatchOrder(c1, none));
            assertTrue(Layout.areContiguousAndMatchOrder(vector, vector));
        }


        @Test
        @DisplayName("isContiguous(order) rejects BOTH/NONE as a query argument")
        void isContiguousRejectsNonCorF() {
            Layout layout = Layout.contiguous(new Shape(2, 3), 1);

            assertThrows(IllegalArgumentException.class,
                    () -> layout.isContiguous(ContiguousOrder.BOTH));
            assertThrows(IllegalArgumentException.class,
                    () -> layout.isContiguous(ContiguousOrder.NONE));
        }
    }


    // =====================================================================================
    // Index arithmetic
    // =====================================================================================
    @Nested
    @DisplayName("slice arithmetic")
    class IndexArithmetic {

        @Test
        @DisplayName("linear element slice is offset + sum(idx * stride)")
        void linearElementIndex() {
            Layout layout = new Layout(new Shape(2, 3, 4), 10, new int[]{12, 4, 1}, 1);

            assertEquals(10, layout.toLinearElementIndex(0, 0, 0));
            assertEquals(10 + 12 + 8 + 3, layout.toLinearElementIndex(1, 2, 3));
        }


        @Test
        @DisplayName("linear element slice ignores itemSize entirely")
        void linearIndexIgnoresItemSize() {
            Shape shape = new Shape(2, 3);
            int[] strides = {3, 1};
            Layout real = new Layout(shape, 4, strides, 1);
            Layout complex = new Layout(shape, 4, strides, 2);

            assertEquals(real.toLinearElementIndex(1, 2), complex.toLinearElementIndex(1, 2));
        }


        @Test
        @DisplayName("buffer slice scales the element slice by itemSize and adds the component")
        void bufferIndexScaling() {
            Layout complex = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 2);
            int element = complex.toLinearElementIndex(1, 2);   // 4 + 3 + 2 = 9

            assertEquals(2*element, complex.toBufferIndex(0, 1, 2), "real component");
            assertEquals(2*element + 1, complex.toBufferIndex(1, 1, 2), "imaginary component");
        }


        @ParameterizedTest
        @ValueSource(ints = {-1, 2, 5})
        @DisplayName("component slice outside [0, itemSize) is rejected")
        void componentIndexBounds(int component) {
            Layout complex = Layout.contiguous(new Shape(2, 3), 2);

            assertThrows(IndexOutOfBoundsException.class,
                    () -> complex.toBufferIndex(component, 0, 0));
        }


        @Test
        @DisplayName("an nD slice whose length differs from the rank is rejected")
        void indexLengthChecked() {
            Layout layout = Layout.contiguous(new Shape(2, 3), 1);

            assertThrows(IllegalArgumentException.class, () -> layout.toLinearElementIndex(0));
            assertThrows(IllegalArgumentException.class, () -> layout.toLinearElementIndex(0, 0, 0));
        }


        @Test
        @DisplayName("a rank-0 layout maps the empty slice to its offset")
        void rankZeroIndexing() {
            Layout scalar = new Layout(new Shape(), 7, new int[0], 1);

            assertEquals(7, scalar.toLinearElementIndex());
            assertEquals(7, scalar.toBufferIndex(0));
        }


        @Test
        @DisplayName("C-contiguous traversal visits slots 0..numel-1 in order")
        void cContiguousEnumerationIsSequential() {
            Layout layout = Layout.contiguous(new Shape(2, 3, 4), 1);
            List<Integer> visited = new ArrayList<>();
            forEachIndex(layout.shape(), idx -> visited.add(layout.toBufferIndex(0, idx)));

            for (int i = 0; i < visited.size(); i++) {
                assertEquals(i, visited.get(i));
            }
        }


        @Test
        @DisplayName("F-contiguous traversal of a matrix is the transpose ordering")
        void fContiguousEnumeration() {
            // (2,3) column-major: element (r,c) lives at slot r + 2c.
            Layout layout = Layout.contiguous(new Shape(2, 3), 1, ContiguousOrder.F);

            assertEquals(0, layout.toBufferIndex(0, 0, 0));
            assertEquals(1, layout.toBufferIndex(0, 1, 0));
            assertEquals(2, layout.toBufferIndex(0, 0, 1));
            assertEquals(5, layout.toBufferIndex(0, 1, 2));
        }
    }


    // =====================================================================================
    // Buffer sizing
    // =====================================================================================
    @Nested
    @DisplayName("buffer sizing")
    class BufferSizing {

        @Test
        @DisplayName("buffer size is numel * itemSize")
        void bufferSize() {
            Layout complex = Layout.contiguous(new Shape(2, 3, 4), 2);

            assertEquals(BigInteger.valueOf(48), complex.bufferSize());
            assertEquals(48, complex.bufferSizeIntValueExact());
        }


        @Test
        @DisplayName("a scalar layout needs itemSize slots")
        void scalarBufferSize() {
            assertEquals(4, Layout.contiguous(new Shape(), 4).bufferSizeIntValueExact());
        }


        @Test
        @DisplayName("an empty layout needs zero slots")
        void emptyBufferSize() {
            assertEquals(0, Layout.contiguous(new Shape(2, 0), 2).bufferSizeIntValueExact());
        }


        @Test
        @DisplayName("overflowing int buffer size throws rather than wrapping")
        void bufferSizeOverflow() {
            Layout layout = Layout.contiguous(new Shape(1 << 30), 8);

            assertThrows(ArithmeticException.class, layout::bufferSizeIntValueExact);
            assertEquals(BigInteger.valueOf(8L << 30), layout.bufferSize());
        }
    }


    // =====================================================================================
    // Reachable extents
    // =====================================================================================
    @Nested
    @DisplayName("minMaxIndex")
    class MinMax {

        // NOTE: these assert the *inclusive last reachable slot* convention, i.e. max is the
        // slice of the final slot of the final element. If you settled on an exclusive end
        // bound instead, adjust the expectations here by +1 -- but keep the brute-force
        // containment test below, which is what actually guards resolveSrc's disjointness check.


        @Test
        @DisplayName("positive strides: min is the offset slot, max the last element's last slot")
        void positiveStrides() {
            Layout layout = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 1);
            var mm = layout.minMaxIndex();

            assertEquals(4, mm.first());
            assertEquals(4 + 3 + 2, mm.second());
        }


        @Test
        @DisplayName("negative strides reach below the offset")
        void negativeStrides() {
            // Rows walk backwards from offset 10.
            Layout layout = new Layout(new Shape(3, 4), 10, new int[]{-4, 1}, 1);
            var mm = layout.minMaxIndex();

            assertEquals(10 - 8, mm.first());
            assertEquals(10 + 3, mm.second());
        }


        @Test
        @DisplayName("bounds are reported in buffer slots, not elements")
        void scaledByItemSize() {
            Layout complex = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 2);
            var mm = complex.minMaxIndex();

            assertEquals(8, mm.first());
            assertTrue(mm.second() >= 2*(4 + 3 + 2),
                    "max must cover the final slot of the final element");
        }


        @ParameterizedTest(name = "{0}")
        @MethodSource("assortedLayouts")
        @DisplayName("every reachable slot lies within [min, max]")
        void containsAllReachableSlots(String name, Layout layout) {
            var mm = layout.minMaxIndex();
            Set<Integer> slots = reachableSlots(layout);

            for (int slot : slots) {
                assertTrue(slot >= mm.first() && slot <= mm.second(),
                        "slot " + slot + " escapes reported extent [" + mm.first() + ", " + mm.second() + "]");
            }
            assertEquals(slots.stream().mapToInt(Integer::intValue).min().orElseThrow(), mm.first(),
                    "min must be tight");
        }


        static Stream<Arguments> assortedLayouts() {
            return Stream.of(
                    Arguments.of("C matrix", Layout.contiguous(new Shape(3, 4), 1)),
                    Arguments.of("F matrix", Layout.contiguous(new Shape(3, 4), 1, ContiguousOrder.F)),
                    Arguments.of("complex C", Layout.contiguous(new Shape(3, 4), 2)),
                    Arguments.of("offset view", new Layout(new Shape(2, 2), 5, new int[]{4, 1}, 1)),
                    Arguments.of("reversed rows", new Layout(new Shape(3, 4), 8, new int[]{-4, 1}, 1)),
                    Arguments.of("gapped", new Layout(new Shape(2, 3), 0, new int[]{10, 2}, 1)),
                    Arguments.of("complex reversed", new Layout(new Shape(3, 2), 6, new int[]{-2, 1}, 2))
            );
        }
    }


    // =====================================================================================
    // Writability
    // =====================================================================================
    @Nested
    @DisplayName("writability")
    class Writability {

        @Test
        @DisplayName("a zero stride over a multi-element axis makes the layout non-writable")
        void zeroStrideNotWritable() {
            Layout layout = new Layout(new Shape(3, 4), 0, new int[]{0, 1}, 1);

            assertFalse(layout.isWritable());
            assertFalse(layout.mayBeInjective());
        }


        @Test
        @DisplayName("a zero stride over an extent-1 axis is harmless")
        void zeroStrideOnSingletonAxis() {
            Layout layout = new Layout(new Shape(1, 4), 0, new int[]{0, 1}, 1);

            assertTrue(layout.isWritable(),
                    "an axis that is never stepped cannot produce a collision");
        }


        @Test
        @DisplayName("interleaved complex layouts are not mistaken for non-overlapping")
        void itemSizeAccountedInOverlap() {
            // Two logical elements one *element* apart are two *slots* apart when itemSize==2.
            // A classifier that forgets itemSize would see stride 1 windows and wrongly accept
            // a layout whose element footprints collide.
            Layout colliding = new Layout(new Shape(2, 3), 0, new int[]{1, 1}, 2);

            assertFalse(colliding.isWritable());
        }


        @ParameterizedTest(name = "{0}")
        @MethodSource("writableLayouts")
        @DisplayName("layouts reported writable are genuinely injective (soundness)")
        void writableImpliesInjective(String name, Layout layout) {
            assumeWritable(layout);
            int expected = layout.shape().numelIntValueExact()*layout.itemSize();

            assertEquals(expected, reachableSlots(layout).size(),
                    "isWritable() must never be true for a layout with aliasing elements");
        }


        static Stream<Arguments> writableLayouts() {
            return Stream.of(
                    Arguments.of("C matrix", Layout.contiguous(new Shape(4, 5), 1)),
                    Arguments.of("F matrix", Layout.contiguous(new Shape(4, 5), 1, ContiguousOrder.F)),
                    Arguments.of("complex C", Layout.contiguous(new Shape(4, 5), 2)),
                    Arguments.of("quaternion C", Layout.contiguous(new Shape(3, 3), 4)),
                    Arguments.of("gapped rows", new Layout(new Shape(3, 4), 0, new int[]{10, 1}, 1)),
                    Arguments.of("reversed rows", new Layout(new Shape(3, 4), 8, new int[]{-4, 1}, 1)),
                    Arguments.of("transposed", new Layout(new Shape(4, 3), 0, new int[]{1, 4}, 1)),
                    Arguments.of("rank-3 permuted", new Layout(new Shape(2, 3, 4), 0, new int[]{1, 8, 2}, 1))
            );
        }


        @Test
        @DisplayName("broadcast layouts are never writable")
        void broadcastNotWritable() {
            Layout row = Layout.contiguous(new Shape(1, 4), 1);
            Layout block = Layout.contiguous(new Shape(3, 4), 1);
            var pair = Layout.broadcast(row, block);

            assertFalse(pair.first().isWritable(),
                    "the broadcast operand repeats values and must not be a write target");
        }
    }


    // =====================================================================================
    // Broadcasting
    // =====================================================================================
    @Nested
    @DisplayName("broadcast")
    class Broadcast {

        @Test
        @DisplayName("identical shapes are returned unchanged")
        void identicalShapesShortCircuit() {
            Layout a = Layout.contiguous(new Shape(2, 3), 1);
            Layout b = Layout.contiguous(new Shape(2, 3), 1, ContiguousOrder.F);
            var pair = Layout.broadcast(a, b);

            assertSame(a, pair.first());
            assertSame(b, pair.second());
        }


        @Test
        @DisplayName("a singleton axis is stretched with a zero stride")
        void singletonAxisStretched() {
            Layout row = Layout.contiguous(new Shape(1, 4), 1);   // strides {4, 1}
            Layout block = Layout.contiguous(new Shape(3, 4), 1); // strides {4, 1}
            var pair = Layout.broadcast(row, block);

            assertEquals(new Shape(3, 4), pair.first().shape());
            assertEquals(new Shape(3, 4), pair.second().shape());
            assertEquals(0, pair.first().stride(0), "stretched axis gets stride 0");
            assertEquals(1, pair.first().stride(1));
            assertArrayEquals(block.strides(), pair.second().strides());
        }


        @Test
        @DisplayName("missing leading axes are prepended with zero strides")
        void rankExpansion() {
            Layout vector = Layout.contiguous(new Shape(4), 1);
            Layout block = Layout.contiguous(new Shape(2, 3, 4), 1);
            var pair = Layout.broadcast(vector, block);

            assertEquals(new Shape(2, 3, 4), pair.first().shape());
            assertArrayEquals(new int[]{0, 0, 1}, pair.first().strides(),
                    "prepended axes must repeat, i.e. carry stride 0");
        }


        @Test
        @DisplayName("both operands may stretch on different axes")
        void mutualStretch() {
            Layout col = Layout.contiguous(new Shape(3, 1), 1);
            Layout row = Layout.contiguous(new Shape(1, 4), 1);
            var pair = Layout.broadcast(col, row);

            assertEquals(new Shape(3, 4), pair.first().shape());
            assertEquals(new Shape(3, 4), pair.second().shape());
            assertEquals(0, pair.first().stride(1));
            assertEquals(0, pair.second().stride(0));
        }


        @Test
        @DisplayName("offsets are preserved through broadcasting")
        void offsetsPreserved() {
            Layout row = new Layout(new Shape(1, 4), 7, new int[]{4, 1}, 1);
            Layout block = new Layout(new Shape(3, 4), 11, new int[]{4, 1}, 1);
            var pair = Layout.broadcast(row, block);

            assertEquals(7, pair.first().offset());
            assertEquals(11, pair.second().offset());
        }


        @Test
        @DisplayName("itemSize is preserved per operand")
        void itemSizePreserved() {
            Layout complexRow = Layout.contiguous(new Shape(1, 4), 2);
            Layout realBlock = Layout.contiguous(new Shape(3, 4), 1);
            var pair = Layout.broadcast(complexRow, realBlock);

            assertEquals(2, pair.first().itemSize());
            assertEquals(1, pair.second().itemSize());
        }


        @ParameterizedTest(name = "{0} vs {1}")
        @MethodSource("incompatiblePairs")
        @DisplayName("incompatible shapes are rejected")
        void incompatibleShapes(int[] a, int[] b) {
            Layout la = Layout.contiguous(new Shape(a), 1);
            Layout lb = Layout.contiguous(new Shape(b), 1);

            assertThrows(IllegalArgumentException.class, () -> Layout.broadcast(la, lb));
        }


        static Stream<Arguments> incompatiblePairs() {
            return Stream.of(
                    Arguments.of(new int[]{3}, new int[]{4}),
                    Arguments.of(new int[]{2, 3}, new int[]{2, 4}),
                    Arguments.of(new int[]{2, 3, 4}, new int[]{3, 3, 4})
            );
        }


        @Test
        @DisplayName("the stretched operand reads the same element along the repeated axis")
        void stretchedOperandRepeatsValues() {
            Layout row = Layout.contiguous(new Shape(1, 4), 1);
            Layout block = Layout.contiguous(new Shape(3, 4), 1);
            Layout stretched = Layout.broadcast(row, block).first();

            for (int c = 0; c < 4; c++) {
                int expected = stretched.toBufferIndex(0, 0, c);
                assertEquals(expected, stretched.toBufferIndex(0, 1, c));
                assertEquals(expected, stretched.toBufferIndex(0, 2, c));
            }
        }
    }


    // =====================================================================================
    // Squeeze
    // =====================================================================================
    @Nested
    @DisplayName("squeeze")
    class Squeeze {

        @Test
        @DisplayName("squeeze() drops singleton axes and their strides together")
        void squeezeAll() {
            Layout layout = new Layout(new Shape(1, 3, 1, 4), 2, new int[]{99, 4, 77, 1}, 1);
            Layout squeezed = layout.squeeze();

            assertEquals(new Shape(3, 4), squeezed.shape());
            assertArrayEquals(new int[]{4, 1}, squeezed.strides());
            assertEquals(2, squeezed.offset());
            assertEquals(1, squeezed.itemSize());
        }


        @Test
        @DisplayName("squeeze() returns this when there is nothing to drop")
        void squeezeIdentity() {
            Layout layout = Layout.contiguous(new Shape(2, 3), 1);

            assertSame(layout, layout.squeeze());
        }


        @Test
        @DisplayName("squeeze(axis) drops only that axis")
        void squeezeSingle() {
            Layout layout = new Layout(new Shape(1, 3, 1), 0, new int[]{3, 1, 1}, 1);

            assertEquals(new Shape(3, 1), layout.squeeze(0).shape());
            assertEquals(new Shape(1, 3), layout.squeeze(2).shape());
            assertSame(layout, layout.squeeze(1));
        }


        @Test
        @DisplayName("squeeze(axes) drops only the requested singleton axes")
        void squeezeMultiple() {
            Layout layout = new Layout(new Shape(1, 3, 1, 1), 0, new int[]{3, 1, 1, 1}, 1);

            assertEquals(new Shape(3, 1), layout.squeeze(0, 2).shape());
            assertArrayEquals(new int[]{1, 1}, layout.squeeze(0, 2).strides());
        }


        @Test
        @DisplayName("squeeze(axes) tolerates repeated axes")
        void squeezeRepeatedAxes() {
            Layout layout = new Layout(new Shape(1, 3), 0, new int[]{3, 1}, 1);

            assertEquals(new Shape(3), layout.squeeze(0, 0).shape());
        }


        @ParameterizedTest(name = "{0}")
        @MethodSource("squeezableLayouts")
        @DisplayName("squeeze preserves the element -> slot mapping")
        void mappingPreserved(String name, Layout layout) {
            Layout squeezed = layout.squeeze();
            List<Integer> before = new ArrayList<>();
            List<Integer> after = new ArrayList<>();

            forEachIndex(layout.shape(), idx -> before.add(layout.toBufferIndex(0, idx)));
            forEachIndex(squeezed.shape(), idx -> after.add(squeezed.toBufferIndex(0, idx)));

            assertEquals(before, after,
                    "squeezing must not change which slot any logical element resolves to");
        }


        static Stream<Arguments> squeezableLayouts() {
            return Stream.of(
                    Arguments.of("leading singleton", new Layout(new Shape(1, 3, 4), 0, new int[]{12, 4, 1}, 1)),
                    Arguments.of("trailing singleton", new Layout(new Shape(3, 4, 1), 2, new int[]{4, 1, 1}, 1)),
                    Arguments.of("interior singleton", new Layout(new Shape(3, 1, 4), 0, new int[]{4, 99, 1}, 1)),
                    Arguments.of("complex", new Layout(new Shape(1, 3, 4), 5, new int[]{12, 4, 1}, 2)),
                    Arguments.of("reversed", new Layout(new Shape(1, 3, 4), 8, new int[]{0, -4, 1}, 1))
            );
        }


        @Test
        @DisplayName("squeezing a contiguous layout leaves it contiguous")
        void contiguityInherited() {
            Layout c = new Layout(new Shape(1, 3, 4), 0, new int[]{12, 4, 1}, 1);
            Layout f = new Layout(new Shape(3, 1, 4), 0, new int[]{1, 99, 3}, 1);

            assertTrue(c.squeeze().isCContiguous());
            assertTrue(f.squeeze().isFContiguous());
        }


        @Test
        @DisplayName("squeezing a writable layout leaves it writable")
        void writabilityInherited() {
            Layout layout = new Layout(new Shape(1, 3, 4), 0, new int[]{12, 4, 1}, 1);

            assertTrue(layout.isWritable());
            assertTrue(layout.squeeze().isWritable());
        }
    }


    // =====================================================================================
    // Value semantics
    // =====================================================================================
    @Nested
    @DisplayName("equals / hashCode")
    class ValueSemantics {

        @Test
        @DisplayName("layouts agreeing on shape, offset, strides, and itemSize are equal")
        void equalLayouts() {
            Layout a = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 2);
            Layout b = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 2);

            assertEquals(a, b);
            assertEquals(b, a);
            assertEquals(a.hashCode(), b.hashCode());
        }


        @Test
        @DisplayName("any differing component breaks equality")
        void unequalLayouts() {
            Layout base = new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 2);

            assertNotEquals(new Layout(new Shape(3, 2), 4, new int[]{3, 1}, 2), base);
            assertNotEquals(new Layout(new Shape(2, 3), 5, new int[]{3, 1}, 2), base);
            assertNotEquals(new Layout(new Shape(2, 3), 4, new int[]{1, 2}, 2), base);
            assertNotEquals(new Layout(new Shape(2, 3), 4, new int[]{3, 1}, 1), base);
        }


        @Test
        @DisplayName("differing itemSize is not collapsed by the hash")
        void itemSizeInHash() {
            Layout real = new Layout(new Shape(2, 3), 0, new int[]{3, 1}, 1);
            Layout complex = new Layout(new Shape(2, 3), 0, new int[]{3, 1}, 2);

            assertNotEquals(real.hashCode(), complex.hashCode());
        }


        @Test
        @DisplayName("reflexive, and unequal to null and other types")
        void basicContract() {
            Layout layout = Layout.contiguous(new Shape(2, 3), 1);

            assertEquals(layout, layout);
            assertNotEquals(null, layout);
            assertNotEquals(layout, layout.shape());
        }


        @Test
        @DisplayName("layouts built by different routes to the same description are equal")
        void constructionRouteIrrelevant() {
            Layout viaFactory = Layout.contiguous(new Shape(2, 3), 2);
            Layout viaConstructor = new Layout(new Shape(2, 3), 0, new int[]{3, 1}, 2);

            assertEquals(viaFactory, viaConstructor);
            assertEquals(viaFactory.hashCode(), viaConstructor.hashCode());
        }
    }


    // =====================================================================================
    // Helpers
    // =====================================================================================


    /// Enumerates every logical nD slice of `shape` in C order, invoking `action` on each.
    /// The array passed to `action` is reused; copy it if you need to retain it.
    private static void forEachIndex(Shape shape, Consumer<int[]> action) {
        int rank = shape.rank();
        int numel = shape.numelIntValueExact();
        if (numel == 0) return;

        int[] idx = new int[rank];
        for (int i = 0; i < numel; i++) {
            action.accept(idx);

            for (int ax = rank - 1; ax >= 0; ax--) {
                if (++idx[ax] < shape.getSize(ax)) break;
                idx[ax] = 0;
            }
        }
    }


    /// Collects every buffer slot reachable through `layout`, across all components of every element.
    private static Set<Integer> reachableSlots(Layout layout) {
        Set<Integer> slots = new HashSet<>();
        forEachIndex(layout.shape(), idx -> {
            for (int c = 0; c < layout.itemSize(); c++) {
                slots.add(layout.toBufferIndex(c, idx));
            }
        });

        return slots;
    }


    /// Guards a soundness test: the layout under test must actually be reported writable, or the
    /// test would pass vacuously.
    private static void assumeWritable(Layout layout) {
        assertTrue(layout.isWritable(),
                "test fixture is not reported writable; the injectivity check would be vacuous. "
                        + "If this fires, the overlap classifier has become too conservative.");
    }
}