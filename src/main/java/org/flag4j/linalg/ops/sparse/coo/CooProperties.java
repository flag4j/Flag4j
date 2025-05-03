/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo;


import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Utility class for computing certain properties of COO matrices. The methods in this class are agnostic to the type of COO matrix.
 */
public final class CooProperties {

    private CooProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a sparse COO matrix is symmetric.
     * @param shape The shape of the COO matrix.
     * @param data Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @param zeroValue Any value in {@code values} equal to {@code zeroValue}
     * will be considered zero and will not be considered when determining symmetry. Equality is determined according to
     * {@link Objects#equals(Object, Object)} where if one of the parameters is {@code null} then the result will always be {@code
     * false}. This means passing {@code zeroValue = null} will result in all items in {@code values} being considered. This is
     * useful if there is no definable zero value for the values of the COO matrix.
     * @return {@code true} if the specified COO matrix is symmetric
     * (i.e., equal to its transpose); {@code false} otherwise.
     */
    public static <T> boolean isSymmetric(Shape shape, T[] data, int[] rowIndices, int[] colIndices, T zeroValue) {
        if(shape.get(0) != shape.get(1)) return false; // Early return for non-square matrix.

        Map<Pair<Integer, Integer>, T> dataMap = new HashMap<Pair<Integer, Integer>, T>();

        for(int i = 0, size=data.length; i < size; i++) {
            if(rowIndices[i] == colIndices[i] || Objects.equals(data[i], zeroValue))
                continue; // This value is zero or on the diagonal. No need to consider.

            var p1 = new Pair<>(rowIndices[i], colIndices[i]);
            var p2 = new Pair<>(colIndices[i], rowIndices[i]);

            if(!dataMap.containsKey(p2)) {
                dataMap.put(p1, data[i]);
            } else if(!dataMap.get(p2).equals(data[i])){
                return false; // Not symmetric.
            } else {
                dataMap.remove(p2);
            }
        }

        // If there are any remaining values a value with the transposed indices was not found in the matrix.
        return dataMap.isEmpty();
    }
}
