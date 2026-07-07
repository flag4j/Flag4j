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


import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooTensor;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.linalg.ops.common.real.RealProperties;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This class contains methods for checking the equality of real sparse tensors.
 */
public final class RealSparseEquals {

    private RealSparseEquals(){
        // Hide default constructor for base class.
        
    }


    /**
     * Checks if two real sparse tensors are real. Assumes the indices of each sparse tensor are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First tensor in the equality check.
     * @param b Second tensor in the equality check.
     * @return True if the tensors are equal. False otherwise.
     */
    public static boolean cooTensorEquals(CooTensor a, CooTensor b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<Double> aEntries = new ArrayList<Double>(a.nnz);
        List<int[]> aIndices = new ArrayList<>(a.nnz);

        List<Double> bEntries = new ArrayList<Double>(b.nnz);
        List<int[]> bIndices = new ArrayList<>(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.data[i] != 0) {
                aEntries.add(a.data[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.data[i] != 0) {
                bEntries.add(b.data[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && Arrays.deepEquals(aIndices.toArray(new int[0][]), bIndices.toArray(new int[0][]));
    }

    /**
     * Checks if two real sparse matrices are real. Assumes the indices of each sparse matrix are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param src1 First matrix in the equality check.
     * @param src2 Second matrix in the equality check.
     * @return True if the matrices are equal. False otherwise.
     */
    public static boolean cooMatrixEquals(CooMatrix src1, CooMatrix src2) {
        if(src1 == null || src2 == null) return false;
        if(src1 == src2) return true;
        if(!src1.shape.equals(src2.shape)) return false;

        // Drop any explicitly stored zeros.
        src1 = src1.dropZeros();
        src2 = src2.dropZeros();

        return Arrays.equals(src1.data, src2.data)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices);
    }



    /**
     * Checks if two real sparse vectors are real. Assumes the indices of each sparse vector are sorted. Any explicitly stored
     * zero's will be ignored.
     * @param a First vector in the equality check.
     * @param b Second vector in the equality check.
     * @return True if the vectors are equal. False otherwise.
     */
    public static boolean cooVectorEquals(CooVector a, CooVector b) {
        // Early returns if possible.
        if(a == b) return true;
        if(a==null || b==null || !a.shape.equals(b.shape)) return false;

        List<Double> aEntries = new ArrayList<Double>(a.nnz);
        List<Integer> aIndices = new ArrayList<>(a.nnz);

        List<Double> bEntries = new ArrayList<Double>(b.nnz);
        List<Integer> bIndices = new ArrayList<>(b.nnz);

        for(int i=0; i<a.nnz; i++) {
            if(a.data[i] != 0) {
                aEntries.add(a.data[i]);
                aIndices.add(a.indices[i]);
            }
        }

        for(int i=0; i<b.nnz; i++) {
            if(b.data[i] != 0) {
                bEntries.add(b.data[i]);
                bIndices.add(b.indices[i]);
            }
        }

        return aEntries.equals(bEntries) && aIndices.equals(bIndices);
    }


    /**
     * Checks that all non-zero items are "close" according to {@link RealProperties#allClose(double[], double[])} and
     *      * all indices are the same.
     * @param src1 First matrix in comparison.
     * @param src2 Second matrix in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all items are "close". Otherwise, false.
     */
    public static boolean allCloseMatrix(CooMatrix src1, CooMatrix src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.rowIndices, src2.rowIndices)
                && Arrays.equals(src1.colIndices, src2.colIndices)
                && RealProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero items are "close" according to {@link RealProperties#allClose(double[], double[], double, double)} and
     * all indices are the same.
     * @param src1 First tensor in comparison.
     * @param src2 Second tensor in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all items are "close". Otherwise, false.
     */
    public static boolean allCloseTensor(CooTensor src1, CooTensor src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.deepEquals(src1.indices, src2.indices)
                && RealProperties.allClose(src1.data, src2.data, relTol, absTol);
    }


    /**
     * Checks that all non-zero items are "close" according to {@link RealProperties#allClose(double[], double[])} and
     * all indices are the same.
     * @param src1 First vector in comparison.
     * @param src2 Second vector in comparison.
     * @param relTol Relative tolerance.
     * @param absTol Absolute tolerance.
     * @return True if all items are "close". Otherwise, false.
     */
    public static boolean allCloseVector(CooVector src1, CooVector src2, double relTol, double absTol) {
        // TODO: We need to first check if values are "close" to zero and remove them. Then do the indices and entry check.
        return src1.shape.equals(src2.shape)
                && Arrays.equals(src1.indices, src2.indices)
                && RealProperties.allClose(src1.data, src2.data, relTol, absTol);
    }
}
