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

package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;

/**
 * This class contains low-level implementations for ops which check if a tensor satisfies some property.
 */
public final class RealDenseProperties {

    private RealDenseProperties() {
        // Hide default constructor.
        
    }


    /**
     * Checks if a real dense matrix is symmetric. That is, if the matrix is equal to its transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is symmetric
     */
    public static boolean isSymmetric(double[] src, Shape shape) {
        // Quick return if possible.
        if(shape.getSize(0)!=shape.getSize(1)) return false;

        int count1, count2, stop;

        for(int i = 0; i<shape.getSize(0); i++) {
            count1 = i*shape.getSize(1);
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != src[count2]) {
                    return false;
                }

                count2+=shape.getSize(1);
            }
        }

        return true;
    }


    /**
     * Checks if a real dense matrix is anti-symmetric. That is, if the matrix is equal to its negative transpose.
     * @param src Entries of the matrix.
     * @param shape Shape of the matrix.
     * @return True if this matrix is anti-symmetric
     */
    public static boolean isAntiSymmetric(double[] src, Shape shape) {
        if(shape.getSize(0)!=shape.getSize(1)) {
            return false;
        }

        int count1;
        int count2;
        int stop;

        for(int i = 0; i<shape.getSize(0); i++) {
            count1 = i*shape.getSize(1);
            count2 = i;
            stop = count1 + i;

            while(count1 < stop) {
                if(src[count1++] != -src[count2]) {
                    return false;
                }

                count2+=shape.getSize(1);
            }
        }

        return true;
    }


    /**
     * Checks if a matrix is the identity matrix exactly.
     * @param src Matrix of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is exactly the identity matrix.
     */
    public static boolean isIdentity(Matrix src) {
        if(src == null || src.numRows!=src.numCols) return false;

        int rows = src.numRows;
        int cols = src.numCols;
        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                if(i==j && src.data[rowOffset + j]!=1) return false;
                if(i!=j &&src.data[rowOffset + j]!=0) return false;
            }
        }

        return true;
    }


    /**
     * Checks if a matrix is the identity matrix approximately. Specifically, if the diagonal data are no farther than
     * 1.001E-5 in absolute value from 1.0 and the non-diagonal data are no larger than 1e-08 in absolute value.
     * These tolerances are derived from the {@link TensorBase#allClose(Object)} method.
     * @param src Matrix of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is exactly the identity matrix.
     */
    public static boolean isCloseToIdentity(Matrix src) {
        if(src == null || src.numRows!=src.numCols) return false;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;

        int rows = src.numRows;
        int cols = src.numCols;
        int pos = 0;
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if((i==j && Math.abs(src.data[pos]-1) > diagTol)
                        || (i!=j && Math.abs(src.data[pos]) > nonDiagTol)) {
                    return false;
                }

                pos++;
            }
        }

        return true;
    }
}
