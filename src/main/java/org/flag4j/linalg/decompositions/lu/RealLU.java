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

package org.flag4j.linalg.decompositions.lu;

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>Instances of this class can be used to compute the LU decomposition of a real dense matrix.
 *
 * <p>The LU decomposition decomposes a matrix <span class="latex-inline">A</span> into the product of
 * a unit-lower triangular matrix <span class="latex-inline">L</span> and an upper triangular matrix
 * <span class="latex-inline">U</span>, such that:
 * <span class="latex-display"><pre>
 *     A = LU</pre></span>
 *
 * <h2>Pivoting Strategies:</h2>
 * <p>Pivoting may be used to improve the stability of the decomposition. Pivoting involves swapping rows and/or columns within the
 * matrix during decomposition.
 *
 * <p>This class supports three pivoting strategies via the {@link Pivoting} enum:
 * <ul>
 *     <li>{@link Pivoting#NONE}: No pivoting is performed. This pivoting strategy is generally <em>not</em> recommended.
 *     <span class="latex-display"><pre>
 *         A = LU</pre></span></li>
 *     <li>{@link Pivoting#PARTIAL}: Only row pivoting is performed to improve numerical stability.
 *     Generally, this is the preferred pivoting strategy. The decomposition then becomes,
 *     <span class="latex-display"><pre>
 *         PA = LU</b></pre></span></li>
 *     where <span class="latex-inline">P</span> is a {@link PermutationMatrix permutation matrix} representing the row swaps.
 *     <li>{@link Pivoting#FULL}: Both row and column pivoting are performed to enhance numerical robustness.
 *     The decomposition then becomes,
 *     <span class="latex-display"><pre>
 *         PAQ = LU</pre></span>
 *     where <span class="latex-inline">P</span> and <span class="latex-inline">Q</span> are 
 *     {@link PermutationMatrix permutation matrices} representing the row and column swaps
 *     respectively.
 *
 *     <p>Full pivoting <em>may</em> be useful for <em>highly</em> ill-conditioned matrices but, for practical
 *     purposes, partial pivoting is generally sufficient and more performant.</li>
 * </ul>
 *
 * <h2>Storage Format:</h2>
 * The computed LU decomposition is stored within a single matrix {@code LU}, where:
 * <ul>
 *     <li>The upper triangular part (including the diagonal) represents the non-zero values of
 *     <span class="latex-inline">U</span>.</li>
 *     <li>The strictly lower triangular part represents the non-zero, non-diagonal values of
 *     <span class="latex-inline">L</span>. Since <span class="latex-inline">L</span> is
 *     unit-lower triangular, the diagonal is not stored as it is known to be all zeros.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code RealLU}.</li>
 *     <li>Call {@link LU#decompose(MatrixMixin)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getL()}, {@link #getU()}, {@link #getP()}, and {@link #getQ()}.</li>
 * </ol>
 *
 * @see ComplexLU
 * @see FieldLU
 * @see Pivoting
 * @see PermutationMatrix
 * @see Matrix
 */
public class RealLU extends LU<Matrix> {
    
    
    /**
     * <p>Constructs a LU decomposer for real dense matrices.
     * <p>This decomposition will be performed out-of-place using partial pivoting.
     */
    public RealLU() {
        super(Pivoting.PARTIAL, false);
    }


    /**
     * <p>Constructs a LU decomposer for real dense matrices.
     * <p>This decomposition will be performed out-of-place.
     * @param pivoting Pivoting to use. If pivoting is 2, full pivoting will be used. If pivoting is 1, partial pivoting
     *                 will be used. If pivoting is any other value, no pivoting will be used.
     */
    public RealLU(Pivoting pivoting) {
        super(pivoting, false);
    }


    /**
     * Constructs a LU decomposer for real dense matrices.
     * @param pivoting Pivoting to use.
     * @param inPlace Flag indicating if the decomposition should be done in/out-of-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in-place.</li>
     *     <li>If {@code true}, then the decomposition will be done out-of-place.</li>
     * </ul>
     */
    protected RealLU(Pivoting pivoting, boolean inPlace) {
        super(pivoting, inPlace);
    }


    /**
     * Computes the LU decomposition using no pivoting (i.e., rows and columns are not swapped).
     */
    @Override
    protected void noPivot() {
        int colStop = Math.min(LU.numCols, LU.numRows);

        // Using Gaussian elimination and no pivoting
        for(int j=0; j<colStop; j++) {
            if(j<LU.numRows && Math.abs(LU.data[j*LU.numCols + j]) == 0.0)
                throw new LinearAlgebraException(ZERO_PIV_ERR);

            computeRows(j);
        }
    }


    /**
     * Computes the LU decomposition using partial pivoting (i.e., row swapping).
     */
    @Override
    protected void partialPivot() {
        int colStop = Math.min(LU.numCols, LU.numRows);
        int maxIndex;

        // Using Gaussian elimination with row pivoting.
        for(int j=0; j<colStop; j++) {
            maxIndex = maxColIndex(j); // Find row index of max value (in absolute value) in column j so that the index >= j.

            // Make the appropriate swaps in LU and P (This is the partial pivoting step).
            if(j!=maxIndex && maxIndex>=0)
                swapRows(j, maxIndex);

            // Check for zero pivot after swapping.
            if (j < LU.numRows && LU.data[j*LU.numCols + j] == 0.0)
                throw new LinearAlgebraException(ZERO_PIV_ERR);

            computeRows(j);
        }
    }


    /**
     * Computes the LU decomposition using full/rook pivoting (i.e., row and column swapping).
     */
    @Override
    protected void fullPivot() {
        int[] maxIndex;
        int colStop = Math.min(LU.numCols, LU.numRows);

        // Using Gaussian elimination with row and column (rook) pivoting.
        for(int j=0; j<colStop; j++) {
            maxIndex = maxIndex(j);

            // Make the appropriate swaps in LU, P and Q (This is the full pivoting step).
            if(j!=maxIndex[0] && maxIndex[0]!=-1)
                swapRows(j, maxIndex[0]);
            if(j!=maxIndex[1] && maxIndex[1]!=-1)
                swapCols(j, maxIndex[1]);

            // Check for zero pivot after both row and column swaps.
            if (j < LU.numRows && LU.data[j * LU.numCols + j] == 0.0)
                throw new LinearAlgebraException(ZERO_PIV_ERR);

            computeRows(j);
        }
    }


    /**
     * Helper method which computes rows in the gaussian elimination algorithm.
     * @param j Column for which to compute values to the right of.
     */
    private void computeRows(int j) {
        double m;
        int pivotRow = j*LU.numCols;
        int iRow;

        for(int i=j+1; i<LU.numRows; i++) {
            iRow = i*LU.numCols;
            m = LU.data[iRow + j];
            m = LU.data[pivotRow + j] == 0 ? m : m/LU.data[pivotRow + j];

            if(m!=0) {
                // Compute and set U values.
                for (int k = j; k < LU.numCols; k++)
                    LU.data[iRow + k] -= m * LU.data[pivotRow + k];
            }

            // Compute and set L value.
            LU.data[iRow + j] = m;
        }
    }


    /**
     * Computes the max absolute value in a column so that the row is >= j.
     * @param j column index.
     * @return The index of the maximum absolute value in the specified column such that the row is >= j.
     */
    private int maxColIndex(int j) {
        int maxIndex = -1;
        double currentMax = -1;
        double value;

        for(int i=j; i<LU.numRows; i++) {
            value = Math.abs(LU.data[i*LU.numCols + j]);
            if(value > currentMax) {
                currentMax = value;
                maxIndex = i;
            }
        }

        return maxIndex;
    }


    /**
     * Computes maximum absolute value in sub portion of LU matrix below and right of {@code startIndex, startIndex}.
     * @param startIndex Top left index of row and column of sub matrix.
     * @return Index of maximum absolute value in specified sub matrix.
     */
    private int[] maxIndex(int startIndex) {
        double currentMax = -1;
        double value;
        int rowIdx;
        int[] index = {-1, -1};

        for(int i=startIndex; i<LU.numRows; i++) {
            rowIdx = i*LU.numCols;

            for(int j=startIndex; j<LU.numCols; j++) {
                value = Math.abs(LU.data[rowIdx + j]);

                if(value > currentMax) {
                    currentMax = value;
                    index[0] = i;
                    index[1] = j;
                }
            }
        }

        return index;
    }


    /**
     * Gets the unit lower triangular matrix of the decomposition.
     *
     * @return The lower triangular matrix of the decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(Matrix)}.
     */
    @Override
    public Matrix getL() {
        ensureHasDecomposed();
        Matrix L = new Matrix(LU.numRows, Math.min(LU.numRows, LU.numCols));

        // Copy L values from LU matrix.
        for(int i=0; i<LU.numRows; i++) {
            if(i<LU.numCols) L.data[i*L.numCols + i] = 1; // Set principle diagonal to be 1, so it is unit lower-triangular.
            System.arraycopy(LU.data, i*LU.numCols, L.data, i*L.numCols, i);
        }

        return L;
    }


    /**
     * Gets the upper triangular matrix of the decomposition.
     *
     * @return The lower triangular matrix of the decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(Matrix)}.
     */
    @Override
    public Matrix getU() {
        ensureHasDecomposed();
        Matrix U = new Matrix(Math.min(LU.numRows, LU.numCols), LU.numCols);

        int stopIdx = Math.min(LU.numRows, LU.numCols);

        // Copy U values from LU matrix.
        for(int i=0; i<stopIdx; i++) {
            System.arraycopy(LU.data, i*(LU.numCols+1),
                    U.data, i*(LU.numCols+1), LU.numCols-i);
        }

        return U;
    }
}
