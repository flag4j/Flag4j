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

package org.flag4j.linalg.ops.sparse.csr.real_complex;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CooCVector;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ValidateParameters;

import java.util.*;

/**
 * This class provides low-level implementation of matrix multiplication between a real CSR matrix and a complex
 * CSR matrix.
 */
public final class RealComplexCsrMatMult {

    private RealComplexCsrMatMult() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrix standard(CsrMatrix src1, CsrCMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        Complex128[] destEntries = new Complex128[src1.numRows*src2.numCols];
        Arrays.fill(destEntries, Complex128.ZERO);

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;
            int stop = src1.rowPointers[i+1];

            for(int aIndex=src1.rowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];
                int innerStop = src2.rowPointers[aCol+1];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    Complex128 bVal = src2.data[bIndex];

                    destEntries[rowOffset + bCol] = destEntries[rowOffset + bCol].add(bVal.mult(aVal));
                }
            }
        }

        return new CMatrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices. The result is a dense matrix.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Entries of the dense matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CMatrix standard(CsrCMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        Complex128[] destEntries = new Complex128[src1.numRows*src2.numCols];
        Arrays.fill(destEntries, Complex128.ZERO);

        for(int i=0; i<src1.numRows; i++) {
            int rowOffset = i*src2.numCols;
            int stop = src1.rowPointers[i+1];

            for(int aIndex=src1.rowPointers[i]; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                Complex128 aVal = src1.data[aIndex];
                int innerStop = src2.rowPointers[aCol+1];

                for(int bIndex=src2.rowPointers[aCol]; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.data[bIndex];

                    destEntries[rowOffset + bCol] = destEntries[rowOffset + bCol].add(aVal.mult(bVal));
                }
            }
        }

        return new CMatrix(new Shape(src1.numRows, src2.numCols), destEntries);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrMatrix, CsrCMatrix)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrCMatrix standardAsSparse(CsrMatrix src1, CsrCMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<Complex128> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, Complex128> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                double aVal = src1.data[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    Complex128 bVal = src2.data[bIndex];

                    tempMap.merge(bCol, bVal.mult(aVal), Complex128::add);
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

        Complex128[] resultValues = resultList.toArray(new Complex128[0]);
        int[] resultColIndices = ArrayConversions.fromIntegerList(resultColIndexList);

        return new CsrCMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix multiplication between two sparse CSR matrices and returns the result as a sparse matrix. <br>
     *
     * Warning: This method may be slower than {@link #standard(CsrCMatrix, CsrMatrix)}
     * if the result of multiplying this matrix with {@code src2} is not very sparse. Further, multiplying two
     * sparse matrices (even very sparse matrices) may result in a dense matrix so this method should be used with
     * caution.
     * @param src1 First CSR matrix in the multiplication.
     * @param src2 Second CSR matrix in the multiplication.
     * @return Sparse CSR matrix resulting from the matrix multiplication of the two sparse CSR matrices.
     */
    public static CsrCMatrix standardAsSparse(CsrCMatrix src1, CsrMatrix src2) {
        // Ensure matrices have shapes conducive to matrix multiplication.
        ValidateParameters.ensureMatMultShapes(src1.shape, src2.shape);

        int[] resultRowPtr = new int[src1.numRows + 1];
        List<Complex128> resultList = new ArrayList<>();
        List<Integer> resultColIndexList = new ArrayList<>();

        for (int i=0; i<src1.numRows; i++) {
            Map<Integer, Complex128> tempMap = new HashMap<>();
            int start = src1.rowPointers[i];
            int stop = src1.rowPointers[i + 1];

            for (int aIndex=start; aIndex<stop; aIndex++) {
                int aCol = src1.colIndices[aIndex];
                Complex128 aVal = src1.data[aIndex];
                int innerStart = src2.rowPointers[aCol];
                int innerStop = src2.rowPointers[aCol + 1];

                for (int bIndex=innerStart; bIndex<innerStop; bIndex++) {
                    int bCol = src2.colIndices[bIndex];
                    double bVal = src2.data[bIndex];

                    tempMap.merge(bCol, aVal.mult(bVal), Complex128::add);
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

        Complex128[] resultValues = resultList.toArray(new Complex128[0]);
        int[] resultColIndices = ArrayConversions.fromIntegerList(resultColIndexList);

        return new CsrCMatrix(new Shape(src1.numRows, src2.numCols), resultValues, resultRowPtr, resultColIndices);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVector standardVector(CsrMatrix src1, CooCVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);

        Complex128[] destEntries = new Complex128[src1.numRows];
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;

        // Iterate over the non-zero-elements of the sparse vector.
        for (int k = 0; k < src2.data.length; k++) {
            int col = src2.indices[k];
            Complex128 val = src2.data[k];

            // Perform multiplication only for the non-zero-elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];
                Complex128 destVal = destEntries[col];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        double aVal = src1.data[aIndex];
                        destVal = destVal.add(val.mult(aVal));
                    }
                }
            }
        }

        return new CVector(destEntries);
    }


    /**
     * Computes the matrix-vector multiplication between a real sparse CSR matrix and a complex sparse COO vector.
     * @param src1 The matrix in the multiplication.
     * @param src2 Vector in multiplication. Treated as a column vector in COO format.
     * @return The result of the matrix-vector multiplication.
     * @throws IllegalArgumentException If the number of columns in {@code src1} does not equal the number of columns in {@code src2}.
     */
    public static CVector standardVector(CsrCMatrix src1, CooVector src2) {
        // Ensure the matrix and vector have shapes conducive to multiplication.
        ValidateParameters.ensureAllEqual(src1.numCols, src2.size);

        Complex128[] destEntries = new Complex128[src1.numRows];
        Arrays.fill(destEntries, Complex128.ZERO);
        int rows1 = src1.numRows;

        // Iterate over the non-zero-elements of the sparse vector.
        for (int k = 0; k < src2.data.length; k++) {
            int col = src2.indices[k];
            double val = src2.data[k];

            // Perform multiplication only for the non-zero-elements.
            for (int i=0; i<rows1; i++) {
                int start = src1.rowPointers[i];
                int stop = src1.rowPointers[i + 1];
                Complex128 destVal = destEntries[i];

                for (int aIndex=start; aIndex < stop; aIndex++) {
                    int aCol = src1.colIndices[aIndex];
                    if (aCol == col) {
                        Complex128 aVal = src1.data[aIndex];
                        destVal = destVal.add(aVal.mult(val));
                    }
                }
            }
        }

        return new CVector(destEntries);
    }
}
