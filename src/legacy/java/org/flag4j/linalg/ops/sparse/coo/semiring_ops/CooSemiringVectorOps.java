/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo.semiring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * <p>Utility class for computing ops between sparse COO {@link SemiringElement} vectors.
 * <p>Methods in this class assume that non-zero items are sorted lexicographically by index.
 */
public final class CooSemiringVectorOps {

    private CooSemiringVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise vector addition between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the addition. Indices assumed to be sorted lexicographically.
     * @return A items class containing the non-zero items and indices of the sparse COO vector resulting from the vector addition.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends SemiringElement<T>> SparseVectorData<T> add(
            Shape shape1, T[] src1, int[] src1Indices,
            Shape shape2, T[] src2, int[] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        List<T> values = new ArrayList<>(src1.length);
        List<Integer> indices = new ArrayList<>(src1.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.length && src2Counter < src2.length) {
            if(src1Indices[src1Counter] == src2Indices[src2Counter]) {
                values.add(src1[src1Counter].add(src2[src2Counter]));
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1Indices[src1Counter] < src2Indices[src2Counter]) {
                values.add(src1[src1Counter]);
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2[src2Counter]);
                indices.add(src2Indices[src2Counter]);
                src2Counter++;
            }
        }

        // Finish inserting the rest of the values.
        if(src1Counter < src1.length) {
            for(int i=src1Counter; i<src1.length; i++) {
                values.add(src1[i]);
                indices.add(src1Indices[i]);
            }
        } else if(src2Counter < src2.length) {
            for(int i=src2Counter; i<src2.length; i++) {
                values.add(src2[i]);
                indices.add(src2Indices[i]);
            }
        }

        return new SparseVectorData<T>(shape1, values, indices);
    }


    /**
     * Computes the element-wise vector multiplication between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     *
     * @param shape1 Shape of the first vector.
     * @param src1 Non-zero items of the first vector in the element-wise product.
     * @param src1Indices Non-zero indices of the first vector in the element-wise product.
     * @param shape2 Shape of the second vector.
     * @param src2 Non-zero items of the second vector in the element-wise product.
     * @param src2Indices Non-zero indices of the second vector in the element-wise product.
     *
     * @return A items class containing the non-zero items and indices of the sparse COO vector resulting
     * from the element-wise product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends SemiringElement<T>> SparseVectorData<T> elemMult(
            Shape shape1, T[] src1, int[] src1Indices,
            Shape shape2, T[] src2, int[] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        List<T> values = new ArrayList<>(src1.length);
        List<Integer> indices = new ArrayList<>(src1.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.length && src2Counter < src2.length) {
            if(src1Indices[src1Counter]==src2Indices[src2Counter]) {
                // Then indices match, add product of elements.
                values.add(src1[src1Counter].mult(src2[src2Counter]));
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(src1Indices[src1Counter] < src2Indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return new SparseVectorData<T>(shape1, values, indices);
    }


    /**
     * Computes the dot product between two sparse COO vectors.
     * @param shape1 The shape of the first vector.
     * @param src1 The non-zero items of the first vector.
     * @param src1Indices The non-zero indices of the first vector.
     * @param shape2 The shape of the second vector.
     * @param src2 The non-zero items of the second vector.
     * @param src2Indices The non-zero indices of the second vector.
     * @return The result of the dot product between the two specified COO vectors.
     * @throws NDArrayShapeException If {@code !shape1.equals(shape2)}.
     */
    public static <T extends SemiringElement<T>> T dot(Shape shape1, T[] src1, int[] src1Indices,
                                                       Shape shape2, T[] src2, int[] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        T product = null;
        if(src1.length > 0) product = src1[0].getZero();
        else if(src2.length > 0) product = src2[0].getZero();

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.length && src2Counter < src2.length) {
            if(src1Indices[src1Counter]==src2Indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1[src1Counter++].mult(src2[src2Counter++]));
            } else if(src1Indices[src1Counter] < src2Indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }


    /**
     * Computes the vector outer product between two sparse COO vectors.
     * @param src1 Non-zero items of the first vector.
     * @param src1Indices Non-zero indices in the first vector.
     * @param src1Size The full size of the first vector.
     * @param src2 Non-zero items of the second vector.
     * @param src2Indices Non-zero indices of the second vector.
     * @param dest Array to store the dense matrix resulting from the vector outer product.
     */
    public static <T extends SemiringElement<T>> void outerProduct(
            T[] src1, int[] src1Indices, int src1Size,
            T[] src2, int[] src2Indices,
            T[] dest) {

        final SemiringElement<T> zero;
        if(src1.length > 0) zero = src1[0].getZero();
        else if(src2.length > 0) zero = src2[0].getZero();
        else zero = null;

        Arrays.fill(dest, zero);

        int destRow;
        int index1;
        int index2;

        for(int i=0; i<src1.length; i++) {
            index1 = src1Indices[i];
            destRow = index1*src1Size;
            SemiringElement<T> src1Val = src1[i];

            for(int j=0; j<src2.length; j++) {
                index2 = src2Indices[j];
                dest[destRow + index2] = src1Val.mult(src2[j]);
            }
        }
    }
}
