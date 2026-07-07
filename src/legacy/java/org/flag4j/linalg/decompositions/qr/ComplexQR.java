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

package org.flag4j.linalg.decompositions.qr;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;


/**
 * <p>Computes the QR decomposition of dense complex matrix.
 *
 * <p>The QR decomposition factorizes a given matrix <span class="latex-inline">A</span> into the product of an orthogonal 
 * matrix <span class="latex-inline">Q</span>
 * and an upper triangular matrix <span class="latex-inline">R</span>, such that:
 * <span class="latex-display"><pre>
 *     A = QR</pre></span>
 *
 * <p>The decomposition can be computed in either the <em>full</em> or <em>reduced</em> form:
 * <ul>
 *     <li><b>Reduced QR decomposition:</b> When {@code reduced = true}, the decomposition produces a compact form where
 *         <span class="latex-inline">Q</span> has the same number of rows as <span class="latex-inline">A</span> but only as many
 *         columns as the rank of <span class="latex-inline">A</span>.</li>
 *     <li><b>Full QR decomposition:</b> When {@code reduced = false}, the decomposition produces a square orthogonal matrix
 *         <span class="latex-inline">Q</span> with the same number of rows as <span class="latex-inline">A</span>.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * The decomposition workflow typically follows these steps:
 * <ol>
 *     <li>Instantiate an instance of {@code ComplexQR}.</li>
 *     <li>Call {@link #decompose(CMatrix)} to perform the factorization.</li>
 *     <li>Retrieve the resulting matrices using {@link #getQ()} and {@link #getR()}.</li>
 * </ol>
 *
 * @implNote This class extends {@link ComplexUnitaryDecomposition} and provides implementations for computing the
 * QR decomposition efficiently. The decomposition uses Householder transformations to iteratively
 * zero out sub-diagonal entries while maintaining numerical stability.
 *
 * @see RealQR
 * @see #getR()
 * @see #getQ()
 */
public class ComplexQR extends ComplexUnitaryDecomposition {

    /**
     * Flag indicating if the reduced or full decomposition should be computed.
     * <ul>
     *     <li>If {@code true}: the reduced decomposition will be computed.</li>
     *     <li>If {@code false}: the full decomposition will be computed.</li>
     * </ul>
     */
    protected final boolean reduced;


    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #ComplexQR(boolean)
     */
    public ComplexQR() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public ComplexQR(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexQR decompose(CMatrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Creates and initializes <span class="latex-inline">Q</span> to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected CMatrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return CMatrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix, <span class="latex-inline">R</span>, from the last decomposition.
     *
     * @return The upper triangular matrix, <span class="latex-inline">R</span>, from the last decomposition.
     */
    @Override
    public CMatrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix, <span class="latex-inline">R</span>, from the QR decomposition.
     * @return The upper triangular matrix, <span class="latex-inline">R</span>, from the QR decomposition.
     */
    public CMatrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new CMatrix(rRows, numCols));
    }
}
