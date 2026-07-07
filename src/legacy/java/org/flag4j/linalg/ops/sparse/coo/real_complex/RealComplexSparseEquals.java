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

package org.flag4j.linalg.ops.sparse.coo.real_complex;

import org.flag4j.arrays.sparse.*;
import org.flag4j.util.ArrayUtils;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of real/complex sparse tensors.
 */
public final class RealComplexSparseEquals {

    private RealComplexSparseEquals() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a real sparse tensor and complex sparse tensor are equal.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static boolean tensorEquals(CooTensor a, CooCTensor b) {
        return a.shape.equals(b.shape) && ArrayUtils.equals(a.data, b.data)
                && Arrays.deepEquals(a.indices, b.indices);
    }


    /**
     * Checks if a real sparse matrix and complex sparse matrix are equal.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean matrixEquals(CooMatrix a, CooCMatrix b) {
        return a.shape.equals(b.shape) && ArrayUtils.equals(a.data, b.data)
                && Arrays.equals(a.rowIndices, b.rowIndices)
                && Arrays.equals(a.colIndices, b.colIndices);
    }


    /**
     * Checks if a real sparse vector and complex sparse vector are equal.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean vectorEquals(CooVector a, CooCVector b) {
        return a.size==b.size && Arrays.equals(a.indices, b.indices) && ArrayUtils.equals(a.data, b.data);
    }
}
