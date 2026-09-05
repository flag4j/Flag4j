package org.flag4jv3.util.arrays;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

@DisplayName("Array Utils")
public class ArrayUtilsTests {

    @Nested
    @DisplayName("copyOfStridedRange")
    class CopyOfStridedRangeTests {

        @ParameterizedTest
        @MethodSource("validRangeGenerator")
        void validRanges(int[] src, int start, int stop, int stride, int[] exp) {
            assertArrayEquals(exp, ArrayUtils.copyOfStridedRange(src, start, stop, stride));
        }


        static Stream<Arguments> validRangeGenerator() {
            int[] arr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

            return Stream.of(
                    Arguments.of(arr, 4, 9, 1, new int[]{4, 5, 6, 7, 8}),
                    Arguments.of(arr, 4, 9, 3, new int[]{4, 7}),
                    Arguments.of(arr, 0, 5, 2, new int[]{0, 2, 4}),
                    Arguments.of(arr, 5, 12, 2, new int[]{5, 7, 9, 11}),
                    Arguments.of(arr, 5, 12, 20, new int[]{5}),
                    Arguments.of(arr, 9, 4, -1, new int[]{9, 8, 7, 6, 5}),
                    Arguments.of(arr, 9, 4, -3, new int[]{9, 6}),
                    Arguments.of(arr, 5, 0, -2, new int[]{5, 3, 1}),
                    Arguments.of(arr, 11, 5, -2, new int[]{11, 9, 7}),
                    Arguments.of(arr, 11, 5, -20, new int[]{11}),
                    Arguments.of(arr, 5, 5, 1, new int[]{}),
                    Arguments.of(arr, 5, 5, -1, new int[]{}),
                    Arguments.of(new int[]{}, 0, 0, 1, new int[]{})
            );
        }


        @ParameterizedTest
        @MethodSource("invalidRangeGenerator")
        void invalidRanges(int[] src, int start, int stop, int stride) {
            assertThrows(IllegalArgumentException.class, () -> ArrayUtils.copyOfStridedRange(src, start, stop, stride));
        }


        static Stream<Arguments> invalidRangeGenerator() {
            int[] arr = {0, 1, 2, 3, 4, 5};

            return Stream.of(
                    Arguments.of(arr, -1, 9, 1),
                    Arguments.of(arr, 4, 29, 3),
                    Arguments.of(arr, 5, 0, 2),
                    Arguments.of(arr, 0, -1, 2),
                    Arguments.of(arr, 5, 12, 20),
                    Arguments.of(arr, -1, 2, 1),
                    Arguments.of(arr, 0, 5, -2),
                    Arguments.of(arr, 1, 3, 0),
                    Arguments.of(arr, 6, 6, 1)
            );
        }
    }
}
