/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingMatrix;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for computing ops on sparse COO {@link Ring} matrices.
 */
public final class CooRingMatrixOps {

    private CooRingMatrixOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise difference of two sparse matrices. This method assumes that the indices of the two matrices are
     * sorted
     * lexicographically.
     * @param shape1 Shape of the first matrix.
     * @param src1Entries Non-zero data of the first matrix.
     * @param src1RowIndices Non-zero row indices of the first matrix.
     * @param src1ColIndices Non-zero column indices of the first matrix.
     * @param shape2 Shape of the second matrix.
     * @param src2Entries Non-zero data of the second matrix.
     * @param src2RowIndices Non-zero row indices of the second matrix.
     * @param src2ColIndices Non-zero column indices of the second matrix.
     * @return The element-wise difference of the two matrices.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends Ring<V>> SparseMatrixData<V> sub(
            Shape shape1, V[] src1Entries, int[] src1RowIndices, int[] src1ColIndices,
            Shape shape2, V[] src2Entries, int[] src2RowIndices, int[] src2ColIndices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        int initCapacity = Math.max(src1Entries.length, src2Entries.length);

        List<V> diff = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1Entries.length || src2Counter < src2Entries.length) {

            if(src1Counter >= src1Entries.length || src2Counter >= src2Entries.length) {
                add1 = src2Counter >= src2Entries.length;
                add2 = !add1;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]
                    && src1ColIndices[src1Counter] == src2ColIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1ColIndices[src1Counter] < src2ColIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1RowIndices[src1Counter] < src2RowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                diff.add(src1Entries[src1Counter].sub(src2Entries[src2Counter]));
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                diff.add(src1Entries[src1Counter]);
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
            } else {
                diff.add(src2Entries[src2Counter].addInv());
                rowIndices.add(src2RowIndices[src2Counter]);
                colIndices.add(src2ColIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new SparseMatrixData<V>(shape1, diff, rowIndices, colIndices);
    }


    /**
     * Checks if a real sparse matrix is close to the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static <T extends Ring<T>> boolean isCloseToIdentity(AbstractCooRingMatrix<?, ?, ?, T> src) {
        // Ensure the matrix is square and there are the same number of non-zero data as data on the diagonal.
        if(!src.isSquare() || src.data.length < src.numRows) return false;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.E-5;
        double nonDiagTol = 1e-08;

        final T ONE = src.data.length > 0 ? src.data[0].getOne() : null;

        for(int i=0, size=src.data.length; i<size; i++) {
            int row = src.rowIndices[i];
            int col = src.colIndices[i];

            if(row == col && src.data[i].sub(ONE).abs() > diagTol ) {
                return false; // Diagonal value is not close to one.
            } else if(row != col && src.data[i].mag() > nonDiagTol) {
                return false; // Non-diagonal value is not close to zero.
            }
        }

        return true;
    }


    /**
     * Checks if a sparse COO {@link Ring} matrix is Hermitian.
     * @param shape The shape of the COO matrix.
     * @param data Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return {@code true} if the specified COO matrix is Hermitian
     * (i.e., equal to its conjugate transpose); {@code false} otherwise.
     * @param <T> The ring to which the data values of the COO matrix belong.
     */
    public static <T extends Ring<T>> boolean isHermitian(Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        if(shape.getSize(0) != shape.getSize(1)) return false; // Early return for non-square matrix.

        Map<Pair<Integer, Integer>, T> dataMap = new HashMap<Pair<Integer, Integer>, T>();

        for(int i = 0, size=data.length; i < size; i++) {
            if(rowIndices[i] == colIndices[i] || data[i].isZero())
                continue; // This value is zero or on the diagonal. No need to consider.

            var p1 = new Pair<>(rowIndices[i], colIndices[i]);
            var p2 = new Pair<>(colIndices[i], rowIndices[i]);

            if(!dataMap.containsKey(p2)) {
                dataMap.put(p1, data[i]);
            } else if(!dataMap.get(p2).equals(data[i].conj())){
                return false; // Not Hermitian.
            } else {
                dataMap.remove(p2);
            }
        }

        // If there are any remaining values a value with the transposed indices was not found in the matrix.
        return dataMap.isEmpty();
    }
}
