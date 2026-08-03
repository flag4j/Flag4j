package org.flag4jv3.ndarrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.math.BigInteger;
import java.util.Arrays;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/// Tests for [Shape].
///
/// Organized by concern. Where a test asserts a *property* rather than a hard-coded expectation
/// (e.g. "C strides of a shape are the reverse of F strides of the reversed shape"), the property
/// is stated in the display name so a failure tells you which invariant broke.
@DisplayName("Shape")
class ShapeTests {

    // =====================================================================================
    // Construction & validation
    // =====================================================================================
    @Nested
    @DisplayName("construction")
    class Construction {

        @Test
        @DisplayName("rank and dims reflect the constructor arguments")
        void basicAccessors() {
            Shape shape = new Shape(2, 3, 4);

            assertEquals(3, shape.rank());
            assertArrayEquals(new int[]{2, 3, 4}, shape.dims());
            assertEquals(2, shape.getSize(0));
            assertEquals(4, shape.getSize(2));
        }


        @Test
        @DisplayName("rank-0 (scalar) shape is legal and has one element")
        void rankZero() {
            Shape scalar = new Shape();

            assertEquals(0, scalar.rank());
            assertArrayEquals(new int[0], scalar.dims());
            assertEquals(1, scalar.numelIntValueExact());
            assertEquals(BigInteger.ONE, scalar.numel());
        }


        @Test
        @DisplayName("zero-extent dimensions are legal")
        void zeroExtentAllowed() {
            Shape shape = new Shape(2, 0, 3);

            assertEquals(3, shape.rank());
            assertEquals(0, shape.numelIntValueExact());
        }


        @ParameterizedTest(name = "dims = {0}")
        @MethodSource("negativeDims")
        @DisplayName("negative dimensions are rejected")
        void negativeDimsRejected(int[] dims) {
            assertThrows(IllegalArgumentException.class, () -> new Shape(dims));
        }


        static Stream<Arguments> negativeDims() {
            return Stream.of(
                    Arguments.of(new int[]{-1}),
                    Arguments.of(new int[]{2, -3}),
                    Arguments.of(new int[]{2, 3, Integer.MIN_VALUE})
            );
        }


        @Test
        @DisplayName("rank exactly MAX_RANK is accepted")
        void maxRankAccepted() {
            int[] dims = new int[Shape.MAX_RANK];
            Arrays.fill(dims, 1);

            Shape shape = assertDoesNotThrow(() -> new Shape(dims));
            assertEquals(Shape.MAX_RANK, shape.rank());
        }


        @Test
        @DisplayName("rank exceeding MAX_RANK is rejected")
        void aboveMaxRankRejected() {
            int[] dims = new int[Shape.MAX_RANK + 1];
            Arrays.fill(dims, 1);

            assertThrows(IllegalArgumentException.class, () -> new Shape(dims));
        }


        @Test
        @DisplayName("shape does not alias the caller's dims array")
        void dimsDefensivelyCopied() {
            int[] dims = {2, 3, 4};
            Shape shape = new Shape(dims);

            // Prime the numel cache so a later mutation would additionally desync it.
            int numelBefore = shape.numelIntValueExact();
            dims[0] = 99;

            assertArrayEquals(new int[]{2, 3, 4}, shape.dims(),
                    "mutating the caller's array must not change the Shape");
            assertEquals(numelBefore, shape.numelIntValueExact());
        }


        @Test
        @DisplayName("dims() returns a copy the caller cannot use to mutate the shape")
        void dimsAccessorReturnsCopy() {
            Shape shape = new Shape(2, 3);
            int[] view = shape.dims();
            view[0] = 99;

            assertArrayEquals(new int[]{2, 3}, shape.dims());
        }
    }


    // =====================================================================================
    // Element counts
    // =====================================================================================
    @Nested
    @DisplayName("numel")
    class Numel {

        @ParameterizedTest(name = "{0} -> {1} elements")
        @MethodSource("shapesAndCounts")
        @DisplayName("element count is the product of dimensions")
        void counts(int[] dims, int expected) {
            Shape shape = new Shape(dims);

            assertEquals(expected, shape.numelIntValueExact());
            assertEquals(expected, shape.numelLongValueExact());
            assertEquals(BigInteger.valueOf(expected), shape.numel());
            assertTrue(shape.isIntSized());
        }


        static Stream<Arguments> shapesAndCounts() {
            return Stream.of(
                    Arguments.of(new int[0], 1),          // scalar
                    Arguments.of(new int[]{5}, 5),
                    Arguments.of(new int[]{2, 3}, 6),
                    Arguments.of(new int[]{2, 3, 4}, 24),
                    Arguments.of(new int[]{1, 1, 1}, 1),
                    Arguments.of(new int[]{2, 0, 3}, 0),  // empty
                    Arguments.of(new int[]{0}, 0)
            );
        }


        @Test
        @DisplayName("repeated calls return a consistent value (cache is not corrupting)")
        void cacheIsStable() {
            Shape shape = new Shape(3, 4, 5);

            assertEquals(60, shape.numelIntValueExact());
            assertEquals(60, shape.numelIntValueExact());
            assertEquals(60L, shape.numelLongValueExact());
            assertEquals(BigInteger.valueOf(60), shape.numel());
            assertEquals(60, shape.numelIntValueExact());
        }


        @Test
        @DisplayName("int overflow throws, long value still available, isIntSized() is false")
        void intOverflow() {
            // 2^31 elements: overflows int, fits comfortably in long.
            Shape shape = new Shape(1 << 16, 1 << 15);

            assertThrows(ArithmeticException.class, shape::numelIntValueExact);
            assertFalse(shape.isIntSized());
            assertEquals(1L << 31, shape.numelLongValueExact());
            assertEquals(BigInteger.ONE.shiftLeft(31), shape.numel());
        }


        @Test
        @DisplayName("isIntSized() before numelIntValueExact() does not poison the cache")
        void isIntSizedOrderIndependence() {
            Shape a = new Shape(3, 4);
            assertTrue(a.isIntSized());
            assertEquals(12, a.numelIntValueExact());

            Shape b = new Shape(3, 4);
            assertEquals(12, b.numelIntValueExact());
            assertTrue(b.isIntSized());
        }


        @Test
        @DisplayName("long overflow throws from numelLongValueExact")
        void longOverflow() {
            Shape shape = new Shape(1 << 30, 1 << 30, 1 << 30);

            assertThrows(ArithmeticException.class, shape::numelLongValueExact);
            assertEquals(BigInteger.ONE.shiftLeft(90), shape.numel());
        }
    }


    // =====================================================================================
    // Contiguous strides
    // =====================================================================================
    @Nested
    @DisplayName("contiguous strides")
    class ContiguousStrides {

        @ParameterizedTest(name = "{0} -> C strides {1}")
        @MethodSource("cStrideCases")
        @DisplayName("C strides: innermost axis has stride 1, growing right-to-left")
        void cStrides(int[] dims, int[] expected) {
            Shape shape = new Shape(dims);

            assertArrayEquals(expected, shape.getCContiguousStrides());
            assertArrayEquals(expected, shape.getContiguousStrides(ContiguousOrder.C));
        }


        static Stream<Arguments> cStrideCases() {
            return Stream.of(
                    Arguments.of(new int[0], new int[0]),
                    Arguments.of(new int[]{5}, new int[]{1}),
                    Arguments.of(new int[]{2, 3}, new int[]{3, 1}),
                    Arguments.of(new int[]{2, 3, 4}, new int[]{12, 4, 1}),
                    Arguments.of(new int[]{1, 3, 1}, new int[]{3, 1, 1})
            );
        }


        @ParameterizedTest(name = "{0} -> F strides {1}")
        @MethodSource("fStrideCases")
        @DisplayName("F strides: outermost axis has stride 1, growing left-to-right")
        void fStrides(int[] dims, int[] expected) {
            Shape shape = new Shape(dims);

            assertArrayEquals(expected, shape.getFContiguousStrides());
            assertArrayEquals(expected, shape.getContiguousStrides(ContiguousOrder.F));
        }


        static Stream<Arguments> fStrideCases() {
            return Stream.of(
                    Arguments.of(new int[0], new int[0]),
                    Arguments.of(new int[]{5}, new int[]{1}),
                    Arguments.of(new int[]{2, 3}, new int[]{1, 2}),
                    Arguments.of(new int[]{2, 3, 4}, new int[]{1, 2, 6}),
                    Arguments.of(new int[]{1, 3, 1}, new int[]{1, 1, 3})
            );
        }


        @ParameterizedTest(name = "dims = {0}")
        @MethodSource("assortedShapes")
        @DisplayName("duality: C strides of a shape equal reversed F strides of the reversed shape")
        void cfDuality(int[] dims) {
            Shape shape = new Shape(dims);
            Shape reversed = new Shape(reverse(dims));

            assertArrayEquals(shape.getCContiguousStrides(), reverse(reversed.getFContiguousStrides()));
        }


        static Stream<Arguments> assortedShapes() {
            return Stream.of(
                    Arguments.of((Object) new int[]{5}),
                    Arguments.of((Object) new int[]{2, 3}),
                    Arguments.of((Object) new int[]{2, 3, 4}),
                    Arguments.of((Object) new int[]{7, 1, 2, 5})
            );
        }


        @ParameterizedTest
        @ValueSource(strings = {"BOTH", "NONE"})
        @DisplayName("non-C/F orders are rejected")
        void nonCorFRejected(String orderName) {
            Shape shape = new Shape(2, 3);
            ContiguousOrder order = ContiguousOrder.valueOf(orderName);

            assertThrows(RuntimeException.class, () -> shape.getContiguousStrides(order));
        }


        @Test
        @DisplayName("strides overflowing an int throw ArithmeticException")
        void strideOverflow() {
            Shape shape = new Shape(2, 1 << 16, 1 << 16);

            assertThrows(ArithmeticException.class, shape::getCContiguousStrides);
        }


        @Test
        @DisplayName("returned strides are not aliased to any internal state")
        void stridesNotAliased() {
            Shape shape = new Shape(2, 3, 4);
            int[] first = shape.getCContiguousStrides();
            first[0] = 99;

            assertArrayEquals(new int[]{12, 4, 1}, shape.getCContiguousStrides());
        }
    }


    // =====================================================================================
    // Squeeze
    // =====================================================================================
    @Nested
    @DisplayName("squeeze")
    class Squeeze {

        @ParameterizedTest(name = "{0} -> {1}")
        @MethodSource("squeezeAllCases")
        @DisplayName("squeeze() removes every extent-1 axis")
        void squeezeAll(int[] dims, int[] expected) {
            Shape squeezed = new Shape(dims).squeeze();

            assertArrayEquals(expected, squeezed.dims());
        }


        static Stream<Arguments> squeezeAllCases() {
            return Stream.of(
                    Arguments.of(new int[]{1, 3, 1}, new int[]{3}),
                    Arguments.of(new int[]{1, 1, 1}, new int[0]),
                    Arguments.of(new int[]{2, 1, 3, 1, 4}, new int[]{2, 3, 4}),
                    Arguments.of(new int[]{1, 0, 1}, new int[]{0})
            );
        }


        @Test
        @DisplayName("squeeze() returns this when nothing to remove (identity, not copy)")
        void squeezeAllIdentity() {
            Shape shape = new Shape(2, 3, 4);

            assertSame(shape, shape.squeeze());
        }


        @Test
        @DisplayName("squeeze(axis) removes only that axis")
        void squeezeSingleAxis() {
            Shape shape = new Shape(1, 3, 1);

            assertArrayEquals(new int[]{3, 1}, shape.squeeze(0).dims());
            assertArrayEquals(new int[]{1, 3}, shape.squeeze(2).dims());
        }


        @Test
        @DisplayName("squeeze(axis) on a non-singleton axis returns this")
        void squeezeSingleAxisNoOp() {
            Shape shape = new Shape(1, 3, 1);

            assertSame(shape, shape.squeeze(1));
        }


        @Test
        @DisplayName("squeeze(axes) removes only the requested singleton axes")
        void squeezeMultipleAxes() {
            Shape shape = new Shape(1, 3, 1, 1);

            assertArrayEquals(new int[]{3, 1}, shape.squeeze(0, 2).dims());
            assertArrayEquals(new int[]{1, 3}, shape.squeeze(2, 3).dims());
        }


        @Test
        @DisplayName("squeeze(axes) is a no-op when no requested axis is a singleton")
        void squeezeMultipleAxesNoOp() {
            Shape shape = new Shape(2, 3, 4);

            assertSame(shape, shape.squeeze(0, 1, 2));
        }


        @Test
        @DisplayName("squeeze(axes) tolerates repeated axes (idempotent per axis)")
        void squeezeRepeatedAxes() {
            Shape shape = new Shape(1, 3);

            // Naive implementations decrement the squeezed rank once per *occurrence* rather
            // than once per *distinct* axis, undersizing the result array.
            assertArrayEquals(new int[]{3}, shape.squeeze(0, 0).dims());
            assertArrayEquals(new int[]{3}, shape.squeeze(0, 0, 0).dims());
        }


        @ParameterizedTest
        @ValueSource(ints = {-1, 3, 100})
        @DisplayName("out-of-range axes are rejected")
        void invalidAxis(int axis) {
            Shape shape = new Shape(1, 3, 1);

            assertThrows(RuntimeException.class, () -> shape.squeeze(axis));
        }


        @Test
        @DisplayName("squeeze preserves element count")
        void preservesNumel() {
            Shape shape = new Shape(2, 1, 3, 1, 4);

            assertEquals(shape.numelIntValueExact(), shape.squeeze().numelIntValueExact());
        }
    }


    // =====================================================================================
    // Axis permutation
    // =====================================================================================
    @Nested
    @DisplayName("axis permutation")
    class Permutation {

        @Test
        @DisplayName("swapAxes exchanges the two dimensions")
        void swapAxes() {
            Shape shape = new Shape(2, 3, 4);

            assertArrayEquals(new int[]{4, 3, 2}, shape.swapAxes(0, 2).dims());
            assertArrayEquals(new int[]{3, 2, 4}, shape.swapAxes(0, 1).dims());
        }


        @Test
        @DisplayName("swapAxes with equal axes is an identity transform")
        void swapSameAxis() {
            Shape shape = new Shape(2, 3, 4);

            assertEquals(shape, shape.swapAxes(1, 1));
        }


        @Test
        @DisplayName("swapAxes does not mutate the receiver")
        void swapDoesNotMutate() {
            Shape shape = new Shape(2, 3, 4);
            shape.swapAxes(0, 2);

            assertArrayEquals(new int[]{2, 3, 4}, shape.dims());
        }


        @Test
        @DisplayName("permuteAxes reorders dimensions by the given axis order")
        void permuteAxes() {
            Shape shape = new Shape(2, 3, 4);

            assertArrayEquals(new int[]{4, 2, 3}, shape.permuteAxes(2, 0, 1).dims());
            assertArrayEquals(new int[]{2, 3, 4}, shape.permuteAxes(0, 1, 2).dims());
        }


        @Test
        @DisplayName("permuteAxes then inverse permutation round-trips")
        void permuteRoundTrip() {
            Shape shape = new Shape(2, 3, 4, 5);
            int[] perm = {2, 0, 3, 1};
            int[] inverse = new int[perm.length];
            for (int i = 0; i < perm.length; i++) inverse[perm[i]] = i;

            assertEquals(shape, shape.permuteAxes(perm).permuteAxes(inverse));
        }


        @ParameterizedTest(name = "axes = {0}")
        @MethodSource("badPermutations")
        @DisplayName("non-permutations and wrong-length axis lists are rejected")
        void badPermutationsRejected(int[] axes) {
            Shape shape = new Shape(2, 3, 4);

            assertThrows(RuntimeException.class, () -> shape.permuteAxes(axes));
        }


        static Stream<Arguments> badPermutations() {
            return Stream.of(
                    Arguments.of((Object) new int[]{0, 1}),        // too short
                    Arguments.of((Object) new int[]{0, 1, 2, 3}),  // too long
                    Arguments.of((Object) new int[]{0, 0, 1}),     // repeated
                    Arguments.of((Object) new int[]{0, 1, 3})      // out of range
            );
        }
    }


    // =====================================================================================
    // Slice / flatten / isSquare
    // =====================================================================================
    @Nested
    @DisplayName("derived shapes")
    class Derived {

        @Test
        @DisplayName("slice(start) takes the trailing axes")
        void sliceFrom() {
            Shape shape = new Shape(2, 3, 4, 5);

            assertArrayEquals(new int[]{4, 5}, shape.slice(2).dims());
            assertArrayEquals(new int[]{2, 3, 4, 5}, shape.slice(0).dims());
            assertArrayEquals(new int[0], shape.slice(4).dims());
        }


        @Test
        @DisplayName("slice(start, stop) is half-open")
        void sliceRange() {
            Shape shape = new Shape(2, 3, 4, 5);

            assertArrayEquals(new int[]{3, 4}, shape.slice(1, 3).dims());
            assertArrayEquals(new int[0], shape.slice(2, 2).dims());
        }


        @Test
        @DisplayName("slice with out-of-bounds indices throws")
        void sliceOutOfBounds() {
            Shape shape = new Shape(2, 3, 4);

            assertThrows(RuntimeException.class, () -> shape.slice(1, 9));
            assertThrows(RuntimeException.class, () -> shape.slice(2, 1));
        }


        @Test
        @DisplayName("flatten produces a rank-1 shape with the same element count")
        void flatten() {
            Shape shape = new Shape(2, 3, 4);
            Shape flat = shape.flatten();

            assertEquals(1, flat.rank());
            assertEquals(24, flat.getSize(0));
            assertEquals(shape.numelIntValueExact(), flat.numelIntValueExact());
        }


        @Test
        @DisplayName("flatten of a scalar is a single-element vector")
        void flattenScalar() {
            assertArrayEquals(new int[]{1}, new Shape().flatten().dims());
        }


        @Test
        @DisplayName("isSquare is true iff all dimensions are equal")
        void isSquare() {
            assertTrue(new Shape().isSquare());
            assertTrue(new Shape(5).isSquare());
            assertTrue(new Shape(3, 3).isSquare());
            assertTrue(new Shape(3, 3, 3).isSquare());
            assertFalse(new Shape(3, 4).isSquare());
            assertFalse(new Shape(3, 3, 4).isSquare());
        }
    }


    // =====================================================================================
    // Value semantics
    // =====================================================================================
    @Nested
    @DisplayName("equals / hashCode")
    class ValueSemantics {

        @Test
        @DisplayName("equal dimensions imply equality and equal hash codes")
        void equalShapes() {
            Shape a = new Shape(2, 3, 4);
            Shape b = new Shape(2, 3, 4);

            assertEquals(a, b);
            assertEquals(b, a);
            assertEquals(a.hashCode(), b.hashCode());
        }


        @Test
        @DisplayName("differing rank or dimensions imply inequality")
        void unequalShapes() {
            Shape base = new Shape(2, 3, 4);

            assertNotEquals(new Shape(2, 4, 3), base);
            assertNotEquals(new Shape(2, 3), base);
            assertNotEquals(new Shape(2, 3, 4, 1), base);
        }


        @Test
        @DisplayName("extent-1 padding is significant: (3) != (1,3)")
        void paddingIsSignificant() {
            assertNotEquals(new Shape(3), new Shape(1, 3));
        }


        @Test
        @DisplayName("reflexive, and unequal to null and to other types")
        void basicContract() {
            Shape shape = new Shape(2, 3);

            assertEquals(shape, shape);
            assertNotEquals(null, shape);
            assertNotEquals("(2, 3)", shape);
        }


        @Test
        @DisplayName("scalar shapes are equal to each other")
        void scalars() {
            assertEquals(new Shape(), new Shape());
            assertEquals(new Shape().hashCode(), new Shape().hashCode());
        }
    }


    // =====================================================================================
    // Helpers
    // =====================================================================================
    private static int[] reverse(int[] arr) {
        int[] out = new int[arr.length];
        for (int i = 0; i < arr.length; i++) out[i] = arr[arr.length - 1 - i];
        return out;
    }
}