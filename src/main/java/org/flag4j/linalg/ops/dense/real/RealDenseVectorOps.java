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

package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ValidateParameters;


/**
 * This class provides low level implementations for several vector operation.
 */
public final class RealDenseVectorOps {

    /**
     * Minimum number of data to apply concurrent algorithm for outer product.
     */
    private static final int OUTER_CONCURRENT_THRESHOLD = 275_000;

    private RealDenseVectorOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the vector inner product for two real dense vectors.
     * @param src1 Entries of the first vector.
     * @param src2 Entries of the second vector.
     * @return The inner product of the two vectors.
     */
    public static double innerProduct(double[] src1, double[] src2) {
        ValidateParameters.ensureArrayLengthsEq(src1.length, src2.length);
        double innerProd=0;

        for(int i=0, size=src1.length; i<size; i++)
            innerProd += src1[i]*src2[i];

        return innerProd;
    }


    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static double[] outerProduct(double[] src1, double[] src2) {
        int destIndex;
        double[] dest = new double[src1.length*src2.length];

        for(int i=0, size=src1.length; i<size; i++) {
            destIndex = i*src2.length;
            double v1 = src1[i];

            for(double v2 : src2)
                dest[destIndex++] = v1*v2;
        }

        return dest;
    }


    /**
     * Computes the vector outer product between two real dense vectors.
     * @param src1 Entries of first vector.
     * @param src2 Entries of second vector.
     * @return The matrix resulting from the vector outer product.
     */
    public static double[] outerProductConcurrent(double[] src1, double[] src2) {
        double[] dest = new double[src1.length*src2.length];

        ThreadManager.concurrentOperation(src1.length, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int destIndex = i*src2.length;
                double v1 = src1[i];

                for(double v2 : src2)
                    dest[destIndex++] = v1*v2;
            }
        });

        return dest;
    }


    /**
     * Dispatches an outer product problem to an appropriate implementation based on the size of the vectors involved.
     * @param src1 First vector in outer product.
     * @param src2 Second vector in outer product.
     * @return The outer product of the two vectors {@code src1} and {@code src2}.
     */
    public static Matrix dispatchOuter(Vector src1, Vector src2) {
        int totalEntries = src1.size*src2.size;
        if(totalEntries < OUTER_CONCURRENT_THRESHOLD)
            return new Matrix(src1.size, src2.size, outerProduct(src1.data, src2.data));
        else
            return new Matrix(src1.size, src2.size, outerProductConcurrent(src1.data, src2.data));
    }
}
