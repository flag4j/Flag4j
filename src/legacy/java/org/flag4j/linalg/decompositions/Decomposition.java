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

package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.backend.MatrixMixin;


/**
 * <p>An abstract base class representing a matrix decomposition.
 *
 * <p>This class provides a foundation for various matrix decompositions,
 * such as LU, QR, SVD, Schur, and Hessenburg decompositions. Implementing classes must define
 * the decomposition process by overriding the {@link #decompose(MatrixMixin)} method.
 *
 * <h2>Usage:</h2>
 * <p>A typical workflow using a decomposition class follows these steps:
 * <ol>
 *     <li>Instantiate a concrete implementation of a decomposition.</li>
 *     <li>Call {@link #decompose(MatrixMixin)} to perform the decomposition and a specific matrix.</li>
 *     <li>Retrieve decomposition results via additional getter methods provided by the subclass.</li>
 * </ol>
 *
 * <h2>State Management:</h2>
 *
 * <p>The class maintains an internal state flag, {@code hasDecomposed}, which tracks whether
 * a matrix decomposition has been performed. This ensures that methods depending on the
 * decomposition can verify its existence before proceeding. The {@link #ensureHasDecomposed()}
 * method is provided to enforce this check. Below is a minimal example usage of {@link #ensureHasDecomposed()} for
 * a LU decomposition.
 * <pre>{@code
 * class RealLU extends Decomposition<Matrix> {
 *     Matrix L;
 *     Matrix U;
 *     // Other fields and constructors...
 *
 *     @Override
 *     public RealLU decompose(Matrix a) {
 *         // LU implementation...
 *         super.hasDecomposed = true;
 *         return this;
 *     }
 *
 *     public Matrix getL() {
 *         super.ensureHasDecomposed();
 *         return L;
 *     }
 *
 *     public Matrix getU() {
 *         super.ensureHasDecomposed();
 *         return U;
 *     }
 * }
 * }</pre>
 *
 * @param <T> The type of matrix that this decomposition operates on.
 */
public abstract class Decomposition<T extends MatrixMixin<T, ?, ?, ?>> {

    /**
     * Error message to print when a method dependent on the result of the decomposition is called prior to the decomposition
     * being computed.
     */
    private static final String NO_DECOMPOSE_ERR = "No matrix has been decomposed. Must call decompose(...) on this instance first.";

    /**
     * Flag indicating if this instance has computed a decomposition.
     * <ul>
     *     <li>If {@code true} then this instance has decomposed a matrix.</li>
     *     <li>If {@code false} then this instance has <em>not</em> yet decomposed a matrix.</li>
     * </ul>
     */
    protected boolean hasDecomposed = false;


    /**
     * Applies decomposition to the source matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    public abstract Decomposition<T> decompose(T src);


    /**
     * <p>Ensures that this instance has computed a decomposition.
     * <p>This is useful to ensure that a decomposition has been computed in a method that depends on the result of the decomposition.
     * @throws IllegalStateException If {@link #hasDecomposed hasDecomposed == False}.
     */
    protected void ensureHasDecomposed() {
        if(!hasDecomposed)
            throw new IllegalStateException("No matrix has been decomposed. Must call decompose(...) on this instance first.");
    }
}
