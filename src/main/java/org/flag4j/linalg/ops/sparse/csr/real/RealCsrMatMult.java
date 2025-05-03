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

package org.flag4j.linalg.ops.sparse.csr.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;

import java.util.*;

/**
 * This class provides low-level implementation of matrix multiplication between two real CSR matrices.
 */
public final class RealCsrMatMult {

    private RealCsrMatMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static Matrix standard(CsrMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        double[] destEntries = new double[src1.numRows*src2.numCols];

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;

            for(int aIndex=src1.rowPointers[i]; aIndex<src1.rowPointers[i+1]; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<src2.rowPointers[aCol+1]; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.data[bIndex];

                    destEntries[rowOffset + bCol] += aVal*bVal;
                }
            }
        }

        return new Matrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrMatrix, CsrMatrix)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrMatrix standardAsSparse(CsrMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<Double> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, Double> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.data[bIndex];

                    tempMap.merge(bCol, bVal*aVal, Double::sum);
                }
            }

            // Ensure data within each row are sorted by the column indices.
            List<Integer> tempColIndices = new ArrayList<>(tempMap.keySet());
            Collections.sort(tempColIndices);

            for (int colIndex : tempColIndices) {
                resultColIndexList.add(colIndex);
                resultList.add(tempMap.get(colIndex));
            }

            resultRowPtr[i + 1] = resultList.size();
        }

        double[] resultValues = ArrayConversions.fromDoubleList(resultList);
        int[] resultColIndices = ArrayConversions.fromIntegerList(resultColIndexList);

        return new CsrMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a real sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static org.flag4j.arrays.dense.Vector standardVector(CsrMatrix src1, CooVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);

        double[] destEntries = new double[src1.numRows];
        int rows1 = src1.numRows;

        // Iterate over the non-zero-elements of the sparse vector.
        for (int k = 0; k < src2.data.length; k++) {
            int col = src2.indices[k];
            double val = src2.data[k];

            // Perform multiplication only for the non-zero-elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        double aVal = src1.data[aIndex];
                        destEntries[i] += aVal*val;
                    }
                }
            }
        }

        return new Vector(destEntries);
    }
}
