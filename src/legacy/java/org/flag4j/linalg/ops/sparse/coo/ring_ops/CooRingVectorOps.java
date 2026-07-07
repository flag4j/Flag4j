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

package org.flag4j.linalg.ops.sparse.coo.ring_ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseVectorData;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingVector;
import org.flag4j.numbers.RingElement;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.List;

/**
 * Utility class for computing ops on sparse COO {@link RingElement} vectors.
 */
public final class CooRingVectorOps {


    private CooRingVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise vector subtraction between two real sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the difference. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the difference. Indices assumed to be sorted lexicographically.
     * @return A items class containing the non-zero items and indices of the sparse COO vector resulting from the vector subtraction.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends RingElement<T>> SparseVectorData<T> sub(
            Shape shape1, T[] src1, int[] src1Indices,
            Shape shape2, T[] src2, int[] src2Indices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);
        List<T> values = new ArrayList<>(src1.length);
        List<Integer> indices = new ArrayList<>(src1.length);

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.length && src2Counter < src2.length) {
            if(src1Indices[src1Counter] == src2Indices[src2Counter]) {
                values.add(src1[src1Counter].sub(src2[src2Counter]));
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
                src2Counter++;

            } else if(src1Indices[src1Counter] < src2Indices[src2Counter]) {
                values.add(src1[src1Counter]);
                indices.add(src1Indices[src1Counter]);
                src1Counter++;
            } else {
                values.add(src2[src2Counter].negate());
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
                values.add(src2[i].negate());
                indices.add(src2Indices[i]);
            }
        }

        return new SparseVectorData<T>(shape1, values, indices);
    }


    /**
     * Computes the inner product of two complex sparse vectors. Both sparse vectors are assumed
     * to have their indices sorted lexicographically.
     * @param src1 First sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @param src2 Second sparse vector in the inner product. Indices assumed to be sorted lexicographically.
     * @return The result of the vector inner product.
     * @throws IllegalArgumentException If the two vectors do not have the same size (full size including zeros).
     */
    public static <T extends RingElement<T>> T inner(
            AbstractCooRingVector<?, ?, ?, ?, T> src1,
            AbstractCooRingVector<?, ?, ?, ?, T> src2) {
        ValidateParameters.ensureEqualShape(src1.shape, src2.shape);

        T product = src1.getZeroElement();

        int src1Counter = 0;
        int src2Counter = 0;

        while(src1Counter < src1.data.length && src2Counter < src2.data.length) {
            if(src1.indices[src1Counter]==src2.indices[src2Counter]) {
                // Then indices match, add product of elements.
                product = product.add(src1.data[src1Counter].mult(src2.data[src2Counter].conj()));
            } else if(src1.indices[src1Counter] < src2.indices[src2Counter]) {
                src1Counter++;
            } else {
                src2Counter++;
            }
        }

        return product;
    }
}
