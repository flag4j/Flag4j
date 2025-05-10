/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains low-level implementations for ops on a real sparse COO vector.
 */
public final class RealCooVectorOps {

    private RealCooVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Adds a real number to each entry of a sparse vector, including the zero data.
     * @param src Sparse vector to add value to.
     * @param a Value to add to the {@code src} sparse vector.
     * @return The result of adding the specified value to the sparse vector.
     */
    public static Vector add(CooVector src, double a) {
        double[] dest = new double[src.size];
        Arrays.fill(dest, a);

        for(int i = 0; i<src.data.length; i++)
            dest[src.indices[i]] += src.data[i];

        return new Vector(dest);
    }


    /**
     * Subtracts a real number from each entry of a sparse vector, including the zero data.
     * @param src Sparse vector to subtract value from.
     * @param a Value to subtract from the {@code src} sparse vector.
     * @return The result of subtracting the specified value from the sparse vector.
     */
    public static Vector sub(CooVector src, double a) {
        double[] dest = new double[src.size];
        Arrays.fill(dest, -a);

        for(int i = 0; i<src.data.length; i++)
            dest[src.indices[i]] += src.data[i];

        return new Vector(dest);
    }


    /**
     * Computes the element-wise vector addition between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return The result of the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooVector add(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);
        List<Double> values = new ArrayList<>(initCapacity);
        List<Integer> indices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter] + src2.data[src2Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2.data[src2Counter]);
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.data.length) {
            for(int i = src1Counter; i<src1.data.length; i++) {
                values.add(src1.data[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.data.length) {
            for(int i = src2Counter; i<src2.data.length; i++) {
                values.add(src2.data[i]);
                indices.add(src2.indices[i]);
            }
        }

        return CooVector.unsafeMake(
                src1.size,
                values.stream().mapToDouble(Double::doubleValue).toArray(),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector subtraction between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the subtraction. Indices assumed to be sorted lexicographically.
     * @return The result of the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooVector sub(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);
        List<Double> values = new ArrayList<>(initCapacity);
        List<Integer> indices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter] == src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter] - src2.data[src2Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                values.add(src1.data[src1Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(-src2.data[src2Counter]);
                indices.add(src2.indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.data.length) {
            for(int i = src1Counter; i<src1.data.length; i++) {
                values.add(src1.data[i]);
                indices.add(src1.indices[i]);
            }
        } else if(src2Counter < src2.data.length) {
            for(int i = src2Counter; i<src2.data.length; i++) {
                values.add(-src2.data[i]);
                indices.add(src2.indices[i]);
            }
        }

        return CooVector.unsafeMake(
                src1.size,
                values.stream().mapToDouble(Double::doubleValue).toArray(),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the element-wise vector multiplication between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the multiplication. Indices assumed to be sorted lexicographically.
     * @return The result of the vector multiplication.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static CooVector elemMult(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        int initCapacity = Math.max(src1.data.length, src2.data.length);
        List<Double> values = new ArrayList<>(initCapacity);
        List<Integer> indices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                values.add(src1.data[src1Counter]*src2.data[src2Counter]);
                indices.add(src1.indices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return CooVector.unsafeMake(
                src1.size,
                values.stream().mapToDouble(Double::doubleValue).toArray(),
                indices.stream().mapToInt(Integer::intValue).toArray()
        );
    }


    /**
     * Computes the inner product of two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static double inner(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);
        double product = 0;
        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product += src1.data[src1Counter]*src2.data[src2Counter];
                src1Counter++;
                src2Counter++;
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the vector outer product between two real sparse vectors.
     * @param src1 Entries of the first sparse vector in the outer product.
     * @param src2 Second sparse vector in the outer product.
     * @return The matrix resulting from the vector outer product.
     */
    public static Matrix outerProduct(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        double[] dest = new double[src2.size*src1.size];
        int destRow;
        int index1;
        int index2;
        int src2Size = src2.data.length;

        for(int i = 0, size = src1.data.length; i<size; i++) {
            index1 = src1.indices[i];
            destRow = index1*src1.size;
            double v1 = src1.data[i];

            for(int j=0; j<src2Size; j++) {
                dest[destRow + src2.indices[j]] = v1*src2.data[j];
            }
        }

        return new Matrix(src1.size, src2.size, dest);
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param src The vector to repeat.
     * @param n Number of times to repeat vector.
     * @param axis Axis along which to repeat vector. If {@code axis=0} then each row of the resulting matrix will be equivalent to
     * this vector. If {@code axis=1} then each column of the resulting matrix will be equivalent to this vector.
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    public static CooMatrix repeat(CooVector src, int n, int axis) {
        ValidateParameters.ensureInRange(axis, 0, 1, "axis");
        ValidateParameters.ensureGreaterEq(0, n, "n");

        Shape tiledShape;
        double[] tiledEntries = new double[n*src.data.length];
        int[] tiledRows = new int[tiledEntries.length];
        int[] tiledCols = new int[tiledEntries.length];
        int nnz = src.nnz;

        if(axis==0) {
            tiledShape = new Shape(n, src.size);

            for(int i=0; i<n; i++) { // Copy values into row and set col indices as vector indices.
                System.arraycopy(src.data, 0, tiledEntries, i*nnz, nnz);
                System.arraycopy(src.indices, 0, tiledCols, i*nnz, src.indices.length);
                Arrays.fill(tiledRows, i*nnz, (i+1)*nnz, i);
            }
        } else {
            int[] colIndices = ArrayBuilder.intRange(0, n);
            tiledShape = new Shape(src.size, n);

            for(int i=0; i<nnz; i++) {
                Arrays.fill(tiledEntries, i*n, (i+1)*n, src.data[i]);
                Arrays.fill(tiledRows, i*n, (i+1)*n, src.indices[i]);
                System.arraycopy(colIndices, 0, tiledCols, i*n, n);
            }
        }

        return CooMatrix.unsafeMake(tiledShape, tiledEntries, tiledRows, tiledCols);
    }



    /**
     * Stacks two vectors along columns as if they were row vectors.
     *
     * @param src1 First vector in the stack.
     * @param src2 Vector to stack to the bottom of the {@code src2} vector.
     * @return The result of stacking this vector and vector {@code src2}.
     * @throws IllegalArgumentException If the number of data in the {@code src1} vector is different from the number of data in
     *                                  the vector {@code src2}.
     */
    public static CooMatrix stack(CooVector src1, CooVector src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        double[] entries = new double[src1.data.length + src2.data.length];
        int[][] indices = new int[2][src1.indices.length + src2.indices.length]; // Row and column indices.

        // Copy values from vector src1.
        System.arraycopy(src1.data, 0, entries, 0, src1.data.length);
        // Copy values from vector src2.
        System.arraycopy(src2.data, 0, entries, src1.data.length, src2.data.length);

        // Set row indices to 1 for src2 values (this vectors row indices are 0 which was implicitly set already).
        Arrays.fill(indices[0], src1.indices.length, entries.length, 1);

        // Copy indices from src1 vector to the column indices.
        System.arraycopy(src1.indices, 0, indices[1], 0, src1.data.length);
        // Copy indices from src2 vector to the column indices.
        System.arraycopy(src2.indices, 0, indices[1], src1.data.length, src2.data.length);

        return CooMatrix.unsafeMake(new Shape(2, src1.size), entries, indices[0], indices[1]);
    }
}

