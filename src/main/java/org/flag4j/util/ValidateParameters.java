/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package org.flag4j.util;

import org.flag4j.arrays.Shape;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * This utility class contains several methods for ensuring parameters satisfy some condition.
 */
public final class ValidateParameters {

    private ValidateParameters() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if two {@link Shape} objects are equivalent.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws ArrayShapeException If shapes are not equivalent.
     */
    public static void ensureEqualShape(Shape shape1, Shape shape2) {
        if(!shape1.equals(shape2))
            throw new ArrayShapeException(ErrorMessages.equalShapeErrMsg(shape1, shape2));
    }


    /**
     * Checks if two {@link Shape} objects satisfy the requirements of matrix-matrix or matrix-vector multiplication.
     * @param shape1 First shape.
     * @param shape2 Second shape.
     * @throws LinearAlgebraException If shapes do not satisfy the requirements of matrix-matrix or matrix-vector multiplication.
     */
    public static void ensureMatMultShapes(Shape shape1, Shape shape2) {
        int r2 = shape2.getRank();

        if (shape1.getRank() != 2 || (r2 != 2 && r2 != 1) || shape1.getSize(1) != shape2.getSize(0))
            throw new LinearAlgebraException(ErrorMessages.matMultShapeErrMsg(shape1, shape2));
    }


    /**
     * Checks that all array lengths are equal.
     * @param lengths An array of array lengths.
     * @throws IllegalArgumentException If all lengths are not equal.
     */
    public static void ensureArrayLengthsEq(int... lengths) {
        for(int i=0; i<lengths.length-1; i++)
            if(lengths[i] != lengths[i+1])
                throw new IllegalArgumentException(ErrorMessages.getArrayLengthsMismatchErr(lengths));
    }


    /**
     * Checks that two shapes have the same total number of data.
     * @param shape1 First shape to compare.
     * @param shape2 Second shape to compare.
     * @throws ArrayShapeException If the two shapes do not have the same total number of data.
     */
    public static void ensureTotalEntriesEqual(Shape shape1, Shape shape2) {
        if(!shape1.totalEntries().equals(shape2.totalEntries()))
            throw new ArrayShapeException(ErrorMessages.getShapeTotalEntriesErr(shape1, shape2));
    }


    /**
     * Checks if arrays have the same number of total data.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total data.
     */
    public static void ensureTotalEntriesEq(Object[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length)
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
    }


    /**
     * Checks if arrays have the same number of total data.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total data.
     */
    public static void ensureTotalEntriesEq(double[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length)
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
    }


    /**
     * Checks if arrays have the same number of total data.
     * @param arr1 First array.
     * @param arr2 Second array.
     * @throws IllegalArgumentException If arrays do not have the same number of total data.
     */
    public static void ensureTotalEntriesEq(int[][] arr1, double[] arr2) {
        if(arr1.length*arr1[0].length != arr2.length)
            throw new IllegalArgumentException(ErrorMessages.getTotalEntriesErr());
    }


    /**
     * Checks if a set of values are all equal.
     * @param values Values to check if they are equal.
     * @throws IllegalArgumentException If any of the specified values are not equal.
     */
    public static void ensureAllEqual(int... values) {
        for(int i=0, size=values.length-1; i<size; i++)
            if(values[i] != values[i+1])
                throw new IllegalArgumentException("Expecting values to be equal but got: " + Arrays.toString(values));
    }


    /**
     * Ensures that a boolean is true.
     * @param expr Boolean to check the truth of.
     * @param errMsg Error message in the case where {@code expr == false}.
     */
    public static void ensureTrue(boolean expr, String errMsg) {
        if(!expr) throw new IllegalArgumentException(errMsg);
    }


    /**
     * Checks that two values are not equal.
     * @param a First value.
     * @param b Second value.
     * @throws IllegalArgumentException If {@code a==b}.
     */
    public static void ensureNotEquals(double a, double b) {
        if(a == b)
            throw new IllegalArgumentException("Expecting values to NOT be equal but got: " + a + ", " + b + ".");
    }


    /**
     * Checks if a set of values is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are less than the threshold.
     */
    public static void ensureAllGreaterEq(double threshold, double... values) {
        for(double value : values) {
            if(value < threshold)
                throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
        }
    }


    /**
     * Checks if a set of values is greater than or equal to a specified threshold.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If any of the values are less than the threshold.
     */
    public static void ensureAllGreaterEq(int threshold, int... values) {
        for(double value : values) {
            if(value<threshold)
                throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
        }
    }


    /**
     * Checks if {@code values[i] >= threshold} for all {@code i = 0, 1, ..., values.length}.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @throws IllegalArgumentException If {@code values[i] < threshold} for <b>any</b> {@code i = 0, 1, ..., values.length}.
     */
    public static void ensureGreaterEq(int threshold, int value) {
        if(value < threshold)
            throw new IllegalArgumentException(ErrorMessages.getGreaterEqErr(threshold, value));
    }


    /**
     * Checks if {@code value >= threshold}.
     * @param threshold Threshold value.
     * @param value Values to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If {@code value < threshold}.
     */
    public static void ensureGreaterEq(double threshold, double value, String name) {
        if(value < threshold)
            throw new IllegalArgumentException(ErrorMessages.getNamedGreaterEqErr(threshold, value, name));
    }


    /**
     * Checks if {@code values[i] <= threshold} for all {@code i = 0, 1, ..., values.length}.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException If {@code values[i] > threshold} for <b>any</b> {@code i = 0, 1, ..., values.length}.
     */
    public static void ensureLessEq(double threshold, double... values) {
        for(double value : values) {
            if(value > threshold)
                throw new IllegalArgumentException(ErrorMessages.getLessEqErr(threshold, value));
        }
    }


    /**
     * Checks if {@code values[i] <= threshold} for all {@code i = 0, 1, ..., values.length}.
     * @param threshold Threshold value.
     * @param values Values to compare against threshold.
     * @throws IllegalArgumentException {@code values[i] > threshold} for <b>any</b> {@code i = 0, 1, ..., values.length}.
     */
    public static void ensureLessEq(int threshold, int... values) {
        for(double value : values) {
            if(value > threshold)
                throw new IllegalArgumentException(ErrorMessages.getLessEqErr(threshold, value));
        }
    }


    /**
     * Checks if {@code values <= threshold}.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If {@code values > threshold}.
     */
    public static void ensureLessEq(double threshold, double value, String name) {
        if(value > threshold) throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
    }


    /**
     * Checks if {@code values <= threshold}.
     * @param threshold Threshold value.
     * @param value Value to compare against threshold.
     * @param name Name of parameter.
     * @throws IllegalArgumentException If {@code values > threshold}.
     */
    public static void ensureLessEq(BigInteger threshold, int value, String name) {
        if(threshold.compareTo(BigInteger.valueOf(value)) < 0)
            throw new IllegalArgumentException(ErrorMessages.getNamedLessEqErr(threshold, value, name));
    }


    /**
     * Checks if {@code value > 0}.
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value <= 0}.
     * @see #ensureNonNegative(int)
     */
    public static void ensurePositive(int value) {
        if(value <= 0) throw new IllegalArgumentException(ErrorMessages.getNonPosErr(value));
    }


    /**
     * Checks if {@code value >= 0}.
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value < 0}.
     * @see #ensurePositive(int)
     */
    public static void ensureNonNegative(int value) {
        if(value < 0) throw new IllegalArgumentException(ErrorMessages.getNegValueErr(value, "value"));
    }


    /**
     * Checks if {@code value >= 0}.
     * @param value Value of interest.
     * @throws IllegalArgumentException If {@code value < 0}.
     * @see #ensurePositive(int)
     */
    public static void ensureNonNegative(int value, String name) {
        if(value < 0) throw new IllegalArgumentException(ErrorMessages.getNegValueErr(value, "value"));
    }


    /**
     * Checks if {@code values} contains only non-negative values.
     * @param values Values of interest.
     * @throws IllegalArgumentException If any element of {@code values} is negative.
     * @see #ensurePositive(int)
     */
    public static void ensureNonNegative(int... values) {
        for(int value : values)
            ensureNonNegative(value);
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param shape Shape to check.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void ensureSquareMatrix(Shape shape) {
        if(shape.getRank()!=2 || shape.getSize(0)!=shape.getSize(1))
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(shape));
    }


    /**
     * Checks if a shape represents a square tensor.
     * @param shape Shape to check.
     * @throws ArrayShapeException If all axes of the shape are not the same length.
     */
    public static void ensureSquare(Shape shape) {
        ValidateParameters.ensureAllEqual(shape.getDims());
    }


    /**
     * Checks if a shape represents a square matrix.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @throws LinearAlgebraException If the shape is not of rank 2 with equal rows and columns.
     */
    public static void ensureSquareMatrix(int numRows, int numCols) {
        if(numRows != numCols)
            throw new LinearAlgebraException(ErrorMessages.getSquareShapeErr(new Shape(numRows, numCols)));
    }


    /**
     * Checks that a shape has the specified rank.
     *
     * @param shape Shape to check.
     * @param expRank Expected rank.
     *
     * @throws LinearAlgebraException If the specified shape does not have the expected rank.
     */
    public static void ensureRank(Shape shape, int expRank) {
        if(shape.getRank() != expRank)
            throw new LinearAlgebraException(ErrorMessages.shapeRankErr(shape.getRank(), expRank));
    }


    /**
     * Checks that an axis is a valid 2D axis. That is, either axis is 0 or 1.
     * @param axis Axis to check.
     * @throws IllegalArgumentException If the axis is not a valid 2D axis.
     */
    public static void ensureAxis2D(int axis) {
        if(!(axis == 0 || axis==1))
            throw new IllegalArgumentException(ErrorMessages.getAxisErr(axis, 0, 1));
    }


    /**
     * Checks that a list of {@code N} axis are a permutation of {@code {0, 1, 2, ..., N-1}}.
     * @param axes List of axes of interest.
     * @throws IllegalArgumentException If {@code axis} is not a permutation of {@code {0, 1, 2, ..., N-1}}.
     */
    public static void ensurePermutation(int... axes) {
        if (axes == null)
            throw new IllegalArgumentException("Array is not a valid permutation.");

        ensurePermutation(axes, axes.length);
    }


    /**
     * Checks that a list of axis are a permutation of {@code {0, 1, 2, ..., n-1}}.
     * @param axes List of axes of interest.
     * @param n The length of the permutation.
     * @throws IllegalArgumentException If {@code axis} is not a permutation of {@code {0, 1, 2, ..., n-1}}.
     */
    public static void ensurePermutation(int[] axes, int n) {
        if (axes == null || axes.length != n)
            throw new IllegalArgumentException("Array length does not match expected size n=" + n);

        boolean[] seen = new boolean[n];
        for (int axis : axes) {
            if(axis < 0 || axis >= n || seen[axis]) {
                throw new IllegalArgumentException("Array is not a permutation of integers 0 through " + (n - 1) +
                        ". Got " + Arrays.toString(axes));
            }
            seen[axis] = true;
        }
    }


    /**
     * Checks that a value is within the specified inclusive range.
     * @param value Value of interest.
     * @param lowerBound Lower bound of range (inclusive).
     * @param upperBound Upper bound of range (inclusive).
     * @param paramName Name of the parameter.
     * @throws IllegalArgumentException If {@code value} is not within the range {@code [lowerBound, upperBound]}
     */
    public static void ensureInRange(double value, double lowerBound, double upperBound, String paramName) {
        if(value < lowerBound || value > upperBound) {
            String name = paramName==null ? "Value" : paramName;
            String errMsg = String.format("Got %s = %f but must be in range [%f, %f]", name, value, lowerBound, upperBound);

            throw new IllegalArgumentException(errMsg);
        }
    }


    /**
     * Checks that a set of indices is within {@code [0, upperBound)}.
     * @param upperBound Upper bound of range for indices (exclusive).
     * @param indices Array of indices to check.
     * @throws IndexOutOfBoundsException If any {@code indices} are not within {@code [0, upperBound)}.
     */
    public static void validateArrayIndices(int upperBound, int... indices) {
        for(int i : indices) {
            if(i < 0 || i >= upperBound) {
                String errMsg = i<0 ?
                        "Index " + i + " is out of bounds for lower bound of 0" :
                        "Index " + i + " is out of bounds for upper bound of " + upperBound + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }


    /**
     * Checks if the provided nD index is contained in a tensor defined by the given {@code shape}.
     * @param shape Shape of the tensor.
     * @param index nD index to check.
     * @throws IndexOutOfBoundsException If {@code index} is not a valid nD index into a tensor
     * of the specified {@code shape}.
     */
    public static void validateTensorIndex(Shape shape, int... index) {
        // TODO: Update to better error messages.
        if(shape.getRank() != index.length) {
            throw new IndexOutOfBoundsException("Expected dimension " + shape.getRank()
                    + " index but got dimension " + index.length + ".");
        }

        for(int i=0, size=index.length; i<size; i++) {
            if(index[i] < 0 || index[i] >= shape.getSize(i)) {
                String errMsg = index[i]<0 ?
                        "dimension " + i + " is out of bounds for lower bound of 0" :
                        "dimension " + i + " is out of bounds for upper bound of " + shape.getSize(i) + ".";

                throw new IndexOutOfBoundsException(errMsg);
            }
        }
    }


    /**
     * Checks that a set of nD indices are valid indices for a tensor with the specified shape.
     * @param shape Shape of the tensor.
     * @param indices Indices to validate.
     */
    public static void validateTensorIndices(Shape shape, int[]... indices) {
        for(int[] index : indices)
            validateTensorIndex(shape, index);
    }


    /**
     * <p>Checks if all provided {@code axes} are valid with respect to the rank of the given {@code shape}.
     * <p>Specifically, an axis is valid if {@code axis >= 0 && axis < shape.getRank()}.
     * @param shape Shape of interest.
     * @param axes Axes to validate.
     * @throws LinearAlgebraException If {@code axis < 0 || axis >= shape.getRank()} for any axis in {@code axes}.
     */
    public static void ensureValidAxes(Shape shape, int... axes) {
        int rank = shape.getRank();

        for(int axis : axes) {
            if(axis < 0 || axis >= rank) {
                throw new LinearAlgebraException(
                        String.format("Axis %d is out of bounds for shape %s with rank %d.", axis, shape, rank));
            }
        }
    }


    /**
     * <p>Checks if all provided {@code axes} are valid with respect to the rank of the given {@code rank}.
     * <p>Specifically, an axis is valid if {@code axis >= 0 && axis < rank}.
     * @param rank Rank of a tensor/shape.
     * @param axes Axes to validate.
     * @throws LinearAlgebraException If {@code axis < 0 || axis >= shape.getRank()} for any axis in {@code axes}.
     */
    public static void ensureValidAxes(int rank, int... axes) {
        for(int axis : axes) {
            if(axis < 0 || axis >= rank)
                throw new LinearAlgebraException(String.format("Axis %d is out of bounds for rank %d.", axis, rank));
        }
    }


    /**
     * Validates that the specified slice [{@code startIdx}, {@code endIdx}) is within the bounds of a vector with the specified
     * length.
     * @param startIdx Staring index of the slice (inclusive).
     * @param endIdx Ending index of the slice (exclusive).
     * @param length The length of the vector to be sliced.
     * @throws IndexOutOfBoundsException If the slice is not within the bounds of the vector.
     * @throws IllegalArgumentException If {@code startIdx >= endIdx} or {@code startIdx < 0}.
     */
    public static void validateVectorSlice(int startIdx, int endIdx, int length) {
        if(startIdx < 0 || startIdx >= endIdx) {
            throw new IllegalArgumentException("Invalid slice. Starting index must be positive and " +
                    "less than the end index but got [" + startIdx + ", " + endIdx + ")");
        }

        int sliceLength = endIdx - startIdx;

        if(endIdx > length) {
            throw new IndexOutOfBoundsException(
                    String.format("The specified slice [%d, %d) of length %d exceeds vector length %d.",
                            startIdx, endIdx, sliceLength, length)
            );
        }
    }
}
