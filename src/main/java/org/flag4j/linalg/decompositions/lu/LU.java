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
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.util.ArrayBuilder;


/**
 * <p>An abstract base class for LU decomposition of a matrix.
 *
 * <p>The LU decomposition decomposes a matrix <span class="latex-simple">A</span> into the product of
 * a unit-lower triangular matrix <span class="latex-simple">L</span> and an upper triangular matrix 
 * <span class="latex-simple">U</span>, such that:
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
 *         PA = LU</pre></span>
 *     </li>
 *     where <span class="latex-simple">P</span> is a {@link PermutationMatrix permutation matrix} representing the row swaps.
 *     <li>{@link Pivoting#FULL}: Both row and column pivoting are performed to enhance numerical robustness.
 *     The decomposition then becomes,
 *     <span class="latex-display"><pre>
 *         PAQ = LU</pre></span>
 *     where <span class="latex-simple">P</span> and <span class="latex-simple">Q</span> are
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
 *     <span class="latex-simple">U</span>.</li>
 *     <li>The strictly lower triangular part represents the non-zero, non-diagonal values of <span class="latex-simple">L</span>.
 *     Since <span class="latex-simple">L</span> is
 *     unit-lower triangular, the diagonal is not stored as it is known to be all zeros.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate a concrete subclass of {@code LU}.</li>
 *     <li>Call {@link #decompose(MatrixMixin)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getL()}, {@link #getU()}, {@link #getP()}, and {@link #getQ()}.</li>
 * </ol>
 *
 * <h2>Implementation Notes:</h2>
 * <p>Subclasses must implement the specific pivoting strategies by defining:
 * <ul>
 *     <li>{@link #noPivot()} - LU decomposition without pivoting.</li>
 *     <li>{@link #partialPivot()} - LU decomposition with row pivoting.</li>
 *     <li>{@link #fullPivot()} - LU decomposition with full pivoting.</li>
 * </ul>
 *
 * @param <T> The type of matrix on which LU decomposition is performed.
 *
 * @see Pivoting
 * @see PermutationMatrix
 */
public abstract class LU<T extends MatrixMixin<T, ?, ?, ?>> extends Decomposition<T> {

    /**
     * Error message for when a zero pivot is encountered.
     */
    protected static final String ZERO_PIV_ERR = "Zero pivot encountered in decomposition LU decomposition.";


    /**
     * Flag indicating what pivoting to use.
     */
    public final Pivoting pivotFlag;
    /**
     * Flag indicating if the decomposition should be done in/out-of-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in-place.</li>
     *     <li>If {@code true}, then the decomposition will be done out-of-place.</li>
     * </ul>
     */
    protected final boolean inPlace;
    /**
     * <p>Storage for <span class="latex-simple">L</span> and <span class="latex-simple">U</span> matrices. Stored in a single matrix.
     *
     * <p>The upper triangular portion of {@code LU}, including the diagonal, stores the non-zero values of
     * <span class="latex-simple">U</span> while the lower
     * triangular portion, excluding the diagonal, stores the non-zero, non-diagonal values of <span class="latex-simple">L</span>.
     * Since <span class="latex-simple">L</span> is unit-lower triangular,
     * the diagonal need not be stored as it is known to be all ones.
     */
    protected T LU;
    /**
     * Permutation matrix to store row swaps if {@link Pivoting#PARTIAL partial pivoting} is used.
     */
    protected PermutationMatrix P;
    /**
     * Permutation matrix to store column swaps if {@link Pivoting#FULL full pivoting} is used.
     */
    protected PermutationMatrix Q;
    /**
     * Tracks the number of row swaps made during partial/full pivoting.
     */
    protected int numRowSwaps;
    /**
     * Tracks the number of column swaps made during full pivoting.
     */
    protected int numColSwaps;
    /**
     * Array to track row swaps made with full/partial pivoting.
     */
    protected int[] rowSwaps;
    /**
     * Array to track column swaps made with full pivoting.
     */
    protected int[] colSwaps;

    /**
     * Constructs a LU decomposer with the specified pivoting.
     * @param pivoting Pivoting to use.
     * @param inPlace Flag indicating if the decomposition should be done in/out-of-place.
     * <ul>
     *     <li>If {@code true}, then the decomposition will be done in-place.</li>
     *     <li>If {@code true}, then the decomposition will be done out-of-place.</li>
     * </ul>
     */
    protected LU(Pivoting pivoting, boolean inPlace) {
        pivotFlag = pivoting;
        this.inPlace = inPlace;
    }


    /**
     * Applies LU decomposition to the source matrix using the pivoting specified in the constructor.
     *
     * @param src The source matrix to decompose. If {@code inPlace} was {@code true} in the constructor then this matrix will be
     * modified.
     * @return A reference to this decomposer.
     */
    @Override
    public LU<T> decompose(T src) {
        initLU(src);
        numRowSwaps = 0; // Set the number of row swaps to zero.
        numColSwaps = 0; // Set the number of row swaps to zero.

        if(pivotFlag == Pivoting.NONE) {
            rowSwaps = colSwaps = null;
            noPivot(); // Compute with no pivoting.
        } else if(pivotFlag == Pivoting.PARTIAL) {
            rowSwaps = ArrayBuilder.intRange(0, LU.numRows());
            partialPivot();
        } else {
            rowSwaps = ArrayBuilder.intRange(0, LU.numRows());
            colSwaps = ArrayBuilder.intRange(0, LU.numCols());
            fullPivot();
        }

        super.hasDecomposed = true;
        return this;
    }


    /**
     * Initializes the {@code LU} matrix by copying the source matrix to decompose.
     * @param src Source matrix to decompose.
     */
    protected void initLU(T src) {
        LU = inPlace ? src : src.copy();
    }


    /**
     * Computes the LU decomposition using no pivoting (i.e., rows and columns are not swapped).
     */
    protected abstract void noPivot();


    /**
     * Computes the LU decomposition using partial pivoting (i.e., row swapping).
     */
    protected abstract void partialPivot();


    /**
     * Computes the LU decomposition using full/rook pivoting (i.e., row and column swapping).
     */
    protected abstract void fullPivot();


    /**
     * Gets the <span class="latex-simple">L</span> and <span class="latex-simple">U</span> matrices of the decomposition combined in a single matrix.
     * @return The <span class="latex-simple">L</span> and <span class="latex-simple">U</span> matrices of the decomposition stored together in a single matrix.
     * The diagonal of <span class="latex-simple">L</span> is all ones and is not stored allowing the diagonal of <span class="latex-simple">U</span> to be stored along
     * the diagonal of the combined matrix.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public T getLU() {
        ensureHasDecomposed();
        return LU;
    }


    /**
     * Gets the unit lower triangular matrix of the decomposition.
     * @return The unit lower triangular matrix of the decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public abstract T getL();


    /**
     * Gets the upper triangular matrix of the decomposition.
     * @return The upper triangular matrix of the decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public abstract T getU();


    /**
     * Gets the row permutation matrix of the decomposition.
     * @return The row permutation matrix of the decomposition. If <em>no</em> pivoting was used, {@code null} will be returned.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public PermutationMatrix getP() {
        ensureHasDecomposed();
        if(rowSwaps != null) P = new PermutationMatrix(rowSwaps);
        else P = null;

        return P;
    }


    /**
     * Gets the column permutation matrix of the decomposition.
     * @return The column permutation matrix of the decomposition. If full pivoting was <em>not</em> used, {@code null} will be
     * returned.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public PermutationMatrix getQ() {
        ensureHasDecomposed();
        // Invert to ensure matrix represents column swaps.
        if(colSwaps != null) Q = new PermutationMatrix(colSwaps).inv();
        else Q = null;

        return Q;
    }


    /**
     * Gets the number of row swaps used in the last decomposition.
     * @return The number of row swaps used in the last decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public int getNumRowSwaps() {
        ensureHasDecomposed();
        return numRowSwaps;
    }


    /**
     * Gets the number of column swaps used in the last decomposition.
     * @return The number of column swaps used in the last decomposition.
     * @throws IllegalStateException If this method is called before {@link #decompose(MatrixMixin)}.
     */
    public int getNumColSwaps() {
        ensureHasDecomposed();
        return numColSwaps;
    }


    /**
     * Tracks the swapping of two rows during gaussian elimination with partial or full pivoting.
     * @param rowIdx1 First row index in swap.
     * @param rowIdx2 Second row index in swap.
     */
    protected void swapRows(int rowIdx1, int rowIdx2) {
        numRowSwaps++;
        int temp = rowSwaps[rowIdx1];
        rowSwaps[rowIdx1] = rowSwaps[rowIdx2];
        rowSwaps[rowIdx2] = temp;

        LU.swapRows(rowIdx1, rowIdx2);
    }


    /**
     * Tracks the swapping of two columns during gaussian elimination with full pivoting.
     * @param colIdx1 First column index in swap.
     * @param colIdx2 Second column index in swap.
     */
    protected void swapCols(int colIdx1, int colIdx2) {
        numColSwaps++;
        int temp = colSwaps[colIdx1];
        colSwaps[colIdx1] = colSwaps[colIdx2];
        colSwaps[colIdx2] = temp;

        LU.swapCols(colIdx1, colIdx2);
    }


    /**
     * Simple enum containing pivoting options for pivoting in LU decomposition.
     */
    public enum Pivoting {
        NONE, PARTIAL, FULL;
    }
}
