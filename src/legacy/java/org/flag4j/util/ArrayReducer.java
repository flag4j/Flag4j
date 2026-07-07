/*
 * MIT License
 *
 * Copyright (c) 2025-2026. Jacob Watters
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

import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;

import java.util.*;
import java.util.function.BinaryOperator;
import java.util.function.DoubleBinaryOperator;
import java.util.function.IntBinaryOperator;
import java.util.function.LongBinaryOperator;


/**
 * A utility class for applying a reduction to an array, possibly along a set of axes of an nD array.
 * This class contains implementations for {@link Object} ndarrays and all primitive type ndarrays.
 */
public final class ArrayReducer {

    // TODO: The nD array reduction methods (that accept a shape and axes parameter) should be moved to the linalg package.
    //  We should consider if all methods should be moved or not... Seems strange to separate these methods into an entirely
    //  different package.
    
    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @param <T> The type of an individual element of the array to reduce.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final <T> T reduce(T[] arr, T identity, BinaryOperator<T> accumulator) {
        if (arr.length == 0)
            return identity;

        T accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.apply(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     *
     * @param shape Shape of the nD array.
     * @param src The elements of the nD array.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @param dest The items array to store the elements of the tensor resulting from the reduction operations. This array must be
     * appropriately sized. Use {@link #getOutSize(Shape, int[]) getOutSize(shape, axes)} to compute the proper size of this array.
     * @param axes The axes over which the reduction is requested.
     * <p>Some special cases:
     * <ul>
     *     <li>If {@code axes.length == 0} then the reduction is applied over <em>no</em> axes and this method effectively copies
     *     the {@code shape} and {@code src} array.</li>
     *     <li>If {@code axes} contains <em>all</em> axes, then the result will be a rank 0 tensor (i.e., a scalar)
     *     which will be the same value returned by {@link #reduce(Object[], Object, BinaryOperator)}.</li>
     * </ul>
     *
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     *
     * @throws NullPointerException If {@code src == null} or if {@code accumulator == null}.
     * @throws IllegalArgumentException If {@code dest} does not have the proper length; exactly equal to the number of elements in
     * the tensor resulting from the reduction operation.
     */
    public static final <T> Pair<Shape, T[]> reduce(
            Shape shape, T[] src, T identity, BinaryOperator<T> accumulator, T[] dest, int... axes) {
        if(axes.length == 0)
            return new Pair<>(shape, src.clone());

        int rank = shape.getRank();
        boolean[] reduceAxes = ShapeUtils.getNormalizedFlags(axes, rank);

        int outRank = rank - axes.length;
        int[] outDims = new int[outRank];
        int[] outProj = new int[rank];
        computeOutDims(shape, reduceAxes, outDims, outProj);

        Shape outShape = new Shape(outDims);
        if(dest.length != outShape.numelIntValueExact()) {
            throw new IllegalArgumentException("dest.length != outShape.totalEntriesIntValueExact");
        }
        if (identity != null) Arrays.fill(dest, identity);

        int[] idx = new int[rank];
        for (int lin = 0, N = src.length; lin < N; ++lin) {

            int proj = 0;
            for (int ax = 0; ax < rank; ++ax)
                proj += idx[ax] * outProj[ax];

            if (identity == null && lin == 0)
                dest[proj] = src[lin];
            else
                dest[proj] = accumulator.apply(dest[proj], src[lin]);

            for (int ax = rank - 1; ax >= 0; --ax) {
                if (++idx[ax] < shape.getDims()[ax]) break;
                idx[ax] = 0;
            }
        }

        return new Pair<>(outShape, dest);
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final boolean reduce(boolean[] arr, Boolean identity, BinaryOperator<Boolean> accumulator) {
        if (arr.length == 0)
            return identity;

        boolean accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.apply(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of a {@link BitSet}, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The {@link BitSet} to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the {@link BitSet}.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code items == null} or if {@code accumulator == null}.
     */
    public static Boolean reduce(BitSet data, Boolean identity, BinaryOperator<Boolean> accumulator) {
        if (data.size() == 0)
            return identity;

        boolean accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = data.get(0);
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size= data.size(); i<size; i++)
            accumulation = accumulator.apply(accumulation, data.get(i));

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     *
     * @param shape Shape of the nD array.
     * @param arr The elements of the nD array.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @param axes The axes over which the reduction is requested.
     * <p>Some special cases:
     * <ul>
     *     <li>If {@code axes.length == 0} then the reduction is applied over <em>no</em> axes and this method effectively copies
     *     the {@code shape} and {@code src} array.</li>
     *     <li>If {@code axes} contains <em>all</em> axes, then the result will be a rank 0 tensor (i.e., a scalar)
     *     which will be the same value returned by {@link #reduce(Object[], Object, BinaryOperator)}.</li>
     * </ul>
     *
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     *
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final Pair<Shape, boolean[]> reduce(
            Shape shape, boolean[] arr, Boolean identity, BinaryOperator<Boolean> accumulator, int... axes) {
        if(axes.length == 0)
            return new Pair<>(shape, arr.clone());

        int rank = shape.getRank();
        boolean[] reduceAxes = ShapeUtils.getNormalizedFlags(axes, rank);

        int outRank = rank - axes.length;
        int[] outDims = new int[outRank];
        int[] outProj = new int[rank];
        computeOutDims(shape, reduceAxes, outDims, outProj);

        Shape outShape = new Shape(outDims);
        boolean[] dst = new boolean[outShape.numelIntValueExact()];
        if (identity != null) Arrays.fill(dst, identity);

        int[] idx = new int[rank];
        for (int lin = 0, N = arr.length; lin < N; ++lin) {

            int proj = 0;
            for (int ax = 0; ax < rank; ++ax)
                proj += idx[ax] * outProj[ax];

            if (identity == null && lin == 0)
                dst[proj] = arr[lin];
            else
                dst[proj] = accumulator.apply(dst[proj], arr[lin]);

            for (int ax = rank - 1; ax >= 0; --ax) {
                if (++idx[ax] < shape.getDims()[ax]) break;
                idx[ax] = 0;
            }
        }

        return new Pair<>(outShape, dst);
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     *
     * @param shape Shape of the nD array.
     * @param arr The elements of the nD array.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @param axes The axes over which the reduction is requested.
     * <p>Some special cases:
     * <ul>
     *     <li>If {@code axes.length == 0} then the reduction is applied over <em>no</em> axes and this method effectively copies
     *     the {@code shape} and {@code src} array.</li>
     *     <li>If {@code axes} contains <em>all</em> axes, then the result will be a rank 0 tensor (i.e., a scalar)
     *     which will be the same value returned by {@link #reduce(Object[], Object, BinaryOperator)}.</li>
     * </ul>
     *
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     *
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final Pair<Shape, BitSet> reduce(
            Shape shape, BitSet arr, Boolean identity, BinaryOperator<Boolean> accumulator, int... axes) {
        if(axes.length == 0)
            return new Pair<>(shape, (BitSet) arr.clone());

        int rank = shape.getRank();
        boolean[] reduceAxes = ShapeUtils.getNormalizedFlags(axes, rank);

        int outRank = rank - axes.length;
        int[] outDims = new int[outRank];
        int[] outProj = new int[rank];
        computeOutDims(shape, reduceAxes, outDims, outProj);

        Shape outShape = new Shape(outDims);
        BitSet dest = new BitSet(outShape.numelIntValueExact());
        if (identity != null) {
            for(int i=0, size=dest.size(); i<size; i++)
                dest.set(i, identity);
        }

        int[] idx = new int[rank];
        for (int lin = 0, N = arr.size(); lin < N; ++lin) {
            int proj = 0;
            for (int ax = 0; ax < rank; ++ax)
                proj += idx[ax] * outProj[ax];

            if (identity == null && lin == 0)
                dest.set(proj, arr.get(lin));
            else
                dest.set(proj, accumulator.apply(dest.get(proj), arr.get(lin)));

            for (int ax = rank - 1; ax >= 0; --ax) {
                if (++idx[ax] < shape.getDims()[ax]) break;
                idx[ax] = 0;
            }
        }

        return new Pair<>(outShape, dest);
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final byte reduce(byte[] arr, Byte identity, BinaryOperator<Byte> accumulator) {
        if (arr.length == 0)
            return identity;

        byte accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.apply(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final short reduce(short[] arr, Short identity, BinaryOperator<Short> accumulator) {
        if (arr.length == 0)
            return identity;

        short accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.apply(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final int reduce(int[] arr, Integer identity, IntBinaryOperator accumulator) {
        if (arr.length == 0)
            return identity;

        int accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.applyAsInt(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final long reduce(long[] arr, Long identity, LongBinaryOperator accumulator) {
        if (arr.length == 0)
            return identity;

        long accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.applyAsLong(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final float reduce(float[] arr, Float identity, BinaryOperator<Float> accumulator) {
        if (arr.length == 0)
            return identity;

        float accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.apply(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     * @param arr The array to reduce.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final double reduce(double[] arr, Double identity, DoubleBinaryOperator accumulator) {
        if (arr.length == 0)
            return identity;

        double accumulation;
        int start;

        if (identity == null) {
            start = 1;
            accumulation = arr[0];
        } else {
            accumulation = identity;
            start = 0;
        }

        for(int i=start, size=arr.length; i<size; i++)
            accumulation = accumulator.applyAsDouble(accumulation, arr[i]);

        return accumulation;
    }


    /**
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     *
     * @param shape Shape of the nD array.
     * @param arr The elements of the nD array.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @param axes The axes over which the reduction is requested.
     * <p>Some special cases:
     * <ul>
     *     <li>If {@code axes.length == 0} then the reduction is applied over <em>no</em> axes and this method effectively copies
     *     the {@code shape} and {@code src} array.</li>
     *     <li>If {@code axes} contains <em>all</em> axes, then the result will be a rank 0 tensor (i.e., a scalar)
     *     which will be the same value returned by {@link #reduce(Object[], Object, BinaryOperator)}.</li>
     * </ul>
     *
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     *
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final Pair<Shape, double[]> reduce(
            Shape shape, double[] arr, Double identity, DoubleBinaryOperator accumulator, int... axes) {
        if(axes.length == 0)
            return new Pair<>(shape, arr.clone());

        int rank = shape.getRank();
        boolean[] reduceAxes = ShapeUtils.getNormalizedFlags(axes, rank);

        int outRank = rank - axes.length;
        int[] outDims = new int[outRank];
        int[] outProj = new int[rank];
        computeOutDims(shape, reduceAxes, outDims, outProj);

        Shape outShape = new Shape(outDims);

        double[] dst = new double[outShape.numelIntValueExact()];
        if (identity != null) Arrays.fill(dst, identity);

        int[] idx = new int[rank];
        for (int lin = 0, N = arr.length; lin < N; ++lin) {

            int proj = 0;
            for (int ax = 0; ax < rank; ++ax)
                proj += idx[ax] * outProj[ax];

            if (identity == null && lin == 0)
                dst[proj] = arr[lin];
            else
                dst[proj] = accumulator.applyAsDouble(dst[proj], arr[lin]);

            for (int ax = rank - 1; ax >= 0; --ax) {
                if (++idx[ax] < shape.getDims()[ax]) break;
                idx[ax] = 0;
            }
        }

        return new Pair<>(outShape, dst);
    }


    /**
     *
     * Reduces the elements of an array, by folding-left, using the specified {@code accumulator} and {@code identity} element.
     *
     * @param shape Shape of the nD array.
     * @param src The elements of the nD array.
     * @param identity The starting element of the reduction. If {@code null}, then the first
     * @param accumulator The binary operator used to accumulate elements of the array.
     * @param dest The items array to store the elements of the tensor resulting from the reduction operations. This array must be
     * appropriately sized. Use {@link #getOutSize(Shape, int[]) getOutSize(shape, axes)} to compute the proper size of this array.
     * @param axes The axes over which the reduction is requested.
     * <p>Some special cases:
     * <ul>
     *     <li>If {@code axes.length == 0} then the reduction is applied over <em>no</em> axes and this method effectively copies
     *     the {@code shape} and {@code src} array.</li>
     *     <li>If {@code axes} contains <em>all</em> axes, then the result will be a rank 0 tensor (i.e., a scalar)
     *     which will be the same value returned by {@link #reduce(Object[], Object, BinaryOperator)}.</li>
     * </ul>
     *
     * @return The scalar resulting from accumulating all elements of the specified array. If the array has length zero, then
     * {@code identity} will be returned, even if it is {@code null}.
     *
     * @param <T> The type of the elements in the array being reduced.
     *
     * @throws NullPointerException If {@code arr == null} or if {@code accumulator == null}.
     */
    public static final <T> Pair<Shape, T[]> reduce(
            Shape shape, T[] src, int[][] indices, T identity,
            BinaryOperator<T> accumulator, T[] dest, int... axes) {

        Objects.requireNonNull(identity, "The identity object must not be null when reducing sparse nD ndarrays.");
        ValidateParameters.ensureValidAxes(shape, axes);

        final int rank = shape.getRank();
        if (axes.length == 0) return new Pair<>(shape, src.clone());  // No axes to reduce; simply return a copy.

        // Determine which axes survive the reduction.
        boolean[] reduceAxis = ShapeUtils.getNormalizedFlags(axes, rank);
        int reducedRank = rank - axes.length;
        int[] reducedDims = new int[reducedRank];
        int[] keepToOrigAxis = new int[reducedRank];

        for (int i = 0, j = 0; i < rank; ++i) {
            if (!reduceAxis[i]) {
                reducedDims[j] = shape.getSize(i);
                keepToOrigAxis[j++] = i;
            }
        }
        Shape reducedShape = new Shape(reducedDims);
        int[] reducedStrides = reducedShape.getStrides();

        // Accumulate values keyed using a flat index.
        Map<Integer, T> accMap = new HashMap<>();
        for (int p = 0, nnz = src.length; p < nnz; ++p) {
            int[] coord = indices[p];

            int flatIdx = 0;
            for (int k = 0; k < reducedRank; ++k)
                flatIdx += coord[keepToOrigAxis[k]]*reducedStrides[k];

            final T value = src[p];
            accMap.compute(flatIdx, (k, oldVal) ->
                    accumulator.apply(oldVal == null ? identity : oldVal, value));
        }

        // Build the sparse COO output tensor.
        int outNnz = accMap.size();
        int[][]  outIdx  = new int[outNnz][reducedRank];

        int q = 0;
        for (Map.Entry<Integer, T> e : accMap.entrySet()) {
            int flat = e.getKey();
            dest[q] = e.getValue();

            // un-flatten the coordinate.
            int rem = flat;
            for (int k = 0; k < reducedRank; k++) {
                int stride = reducedStrides[k];
                int coordinate = rem / stride;
                outIdx[q][k] = coordinate;
                rem -= coordinate * stride;
            }

            q++;
        }

        return new Pair<>(reducedShape, dest);
    }


    /**
     * Computes the output dimensions and projection factors that result in reducing an nD array over the specified axes.
     * @param shape The original shape of the nD array to reduce.
     * @param reduceAxes A list of the axes to reduce over.
     * @param outDims Array to store the output dimensions of the reduced array. Must have length {@code rank - axes.length}.
     * This array will be overwritten with the output dimensions.
     * @param outProj Array to store the projection factors of the reduced array. Must have length {@code rank}.
     * This array will be overwritten with the projection factors.
     */
    private static void computeOutDims(Shape shape, boolean[] reduceAxes, int[] outDims, int[] outProj) {
        int[] dims = shape.getDims();
        int rank = shape.getRank();

        for (int in=0, out=0; in<rank; in++) {
            if (!reduceAxes[in]) {
                outDims[out] = dims[in];
                outProj[in] = (out == 0) ? 1 : outProj[in - 1]*outDims[out - 1];
                out++;
            }
        }
    }


    /**
     * Computes the output dimensions that result in reducing an nD array over the specified axes.
     * @param shape The original shape of the nD array to reduce.
     * @param reduceAxes A list of the axes to reduce over.
     * @param outDims Array to store the output dimensions of the reduced array. Must have length {@code rank - axes.length}.
     * This array will be overwritten with the output dimensions.
     */
    private static void computeOutDims(Shape shape, boolean[] reduceAxes, int[] outDims) {
        int[] dims = shape.getDims();
        int rank = shape.getRank();

        for (int in=0, out=0; in<rank; in++)
            if (!reduceAxes[in])
                outDims[out++] = dims[in];
    }


    /**
     * <p>Computes the number of entries in the tensor resulting from reducing a tensor of shape {@code srcShape}
     * along the specified {@code axes}.
     * <p>This method is intended to be used in conjunction with
     * {@link #reduce(Shape, Object[], Object, BinaryOperator, Object[], int...)} to determine the appropriate size of the
     * {@code dest} array parameter.
     *
     * @param srcShape The shape of the tensor to be reduced.
     * @param axes The axes along which to reduce the output tensor.
     * @return The number of entries in the tensor resulting from reducing a tensor of shape {@code srcShape}
     * along the specified {@code axes}.
     */
    public static int getOutSize(Shape srcShape, int[] axes) {
        if (axes.length == 0)
            return srcShape.numelIntValueExact();

        int rank = srcShape.getRank();
        boolean[] reduceAxes = ShapeUtils.getNormalizedFlags(axes, rank);

        int[] dims = srcShape.getDims();
        int totalEntries = 1;

        for (int in=0, out=0; in<rank; in++)
            if (!reduceAxes[in]) totalEntries *= dims[in];

        return totalEntries;
    }
}
