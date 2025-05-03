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

package org.flag4j.linalg.ops.sparse.coo.real;

import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooMatrix;

import java.util.HashMap;
import java.util.Map;

/**
 * This class contains low level implementations for methods to evaluate certain properties of a real sparse matrix.
 * (i.e., if the matrix is symmetric).
 */
public final class RealSparseMatrixProperties {

    private RealSparseMatrixProperties() {
        // Hide public constructor for utility class.
        
    }


    /**
     * Checks if a real sparse matrix is the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static boolean isIdentity(CooMatrix src) {
        // Ensure the matrix is square and there are at least the same number of non-zero data as data on the diagonal.
        if(!src.isSquare() || src.data.length<src.numRows) return false;

        for(int i = 0, size = src.data.length; i<size; i++) {
            // Ensure value is 1 and on the diagonal.
            if(src.rowIndices[i] != i && src.rowIndices[i] != i && src.data[i] != 1) {
                return false;
            } else if((src.rowIndices[i] != i || src.rowIndices[i] != i) && src.data[i] != 0) {
                return false;
            }
        }

        return true; // If we make it to this point the matrix must be an identity matrix.
    }


    /**
     * Checks if a real sparse matrix is close to the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static boolean isCloseToIdentity(CooMatrix src) {
        // Ensure the matrix is square and there are the same number of non-zero data as data on the diagonal.
        boolean result = src.isSquare() && src.data.length==src.numRows;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;

        for(int i=0, size=src.data.length; i<size; i++) {
            int row = src.rowIndices[i];
            int col = src.colIndices[i];

            if(row == col && Math.abs(src.data[i]-1) > diagTol ) {
                return false; // Diagonal value is not close to one.
            } else if(row != col && Math.abs(src.data[i]) > nonDiagTol) {
                return false; // Non-diagonal value is not close to zero.
            }
        }

        return true;
    }


    /**
     * Checks if a sparse COO matrix is symmetric.
     * @param shape The shape of the COO matrix.
     * @param data Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return {@code true} if the specified COO matrix is symmetric
     * (i.e., equal to its transpose); {@code false} otherwise.
     */
    public static boolean isSymmetric(Shape shape, double[] data, int[] rowIndices, int[] colIndices) {
        if(shape.get(0) != shape.get(1)) return false; // Early return for non-square matrix.

        Map<Pair<Integer, Integer>, Double> dataMap = new HashMap<Pair<Integer, Integer>, Double>();

        for(int i = 0, size=data.length; i < size; i++) {
            if(rowIndices[i] == colIndices[i] || data[i] == 0d)
                continue; // This value is zero or on the diagonal. No need to consider.

            var p1 = new Pair<>(rowIndices[i], colIndices[i]);
            var p2 = new Pair<>(colIndices[i], rowIndices[i]);

            if(!dataMap.containsKey(p2)) {
                dataMap.put(p1, data[i]);
            } else if(dataMap.get(p2) != data[i]){
                return false; // Not symmetric.
            } else {
                dataMap.remove(p2);
            }
        }

        // If there are any remaining values a value with the transposed indices was not found in the matrix.
        return dataMap.isEmpty();
    }


    /**
     * Checks if a sparse COO matrix is symmetric.
     * @param shape The shape of the COO matrix.
     * @param data Non-zero entries of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return {@code true} if the specified COO matrix is symmetric
     * (i.e., equal to its transpose); {@code false} otherwise.
     */
    public static boolean isAntiSymmetric(Shape shape, double[] data, int[] rowIndices, int[] colIndices) {
        if(shape.get(0) != shape.get(1)) return false; // Early return for non-square matrix.

        Map<Pair<Integer, Integer>, Double> dataMap = new HashMap<Pair<Integer, Integer>, Double>();

        for(int i = 0, size=data.length; i < size; i++) {
            if(rowIndices[i] == colIndices[i] || data[i] == 0d)
                continue; // This value is zero or on the diagonal. No need to consider.

            var p1 = new Pair<>(rowIndices[i], colIndices[i]);
            var p2 = new Pair<>(colIndices[i], rowIndices[i]);

            if(!dataMap.containsKey(p2)) {
                dataMap.put(p1, data[i]);
            } else if(dataMap.get(p2) != -data[i]){
                return false; // Not symmetric.
            } else {
                dataMap.remove(p2);
            }
        }

        // If there are any remaining values a value with the transposed indices was not found in the matrix.
        return dataMap.isEmpty();
    }
}







