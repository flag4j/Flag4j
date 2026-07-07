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

package org.flag4j.linalg.solvers;

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;

/**
 * <p>Interface representing a solver for linear systems involving matrices and vectors. Implementations
 * of this interface provide methods to solve equations such as <span class="latex-inline">Ax = b</span> and
 * <span class="latex-inline">AX = B</span>, where <span class="latex-inline">A</span>, <span class="latex-inline">B</span>,
 * and <span class="latex-inline">X</span> are matrices, and <span class="latex-inline">x</span> and
 * <span class="latex-inline">b</span> are vectors.
 *
 * <p>Solvers may compute exact solutions or approximate solutions in a least squares sense, depending on the properties of the system.
 *
 * @param <T> The type of matrices in the linear system, extending {@link MatrixMixin}.
 * @param <U> The type of vectors in the linear system, extending {@link VectorMixin}.
 */
public interface LinearMatrixSolver<T extends MatrixMixin<T, ?, U, ?>,
        U extends VectorMixin<U, T, ?, ?>> extends LinearSolver<T> {


    /**
     * Solves the linear system of equations <span class="latex-inline">Ax = b</span> for the vector
     * <span class="latex-inline">x</span>.
     *
     * @param A The coefficient matrix <span class="latex-inline">A</span> in the linear system.
     * @param b The constant vector in the linear system.
     * @return The solution vector <span class="latex-inline">x</span> satisfying <span class="latex-inline">Ax = b</span>.
     */
    U solve(T A, U b);


    /**
     * Solves the linear matrix equation <span class="latex-inline">AX = B</span> for the matrix <span class="latex-inline">X</span>.
     *
     * @param A The coefficient matrix in the linear system.
     * @param B The constant matrix in the linear system.
     * @return The solution matrix <span class="latex-inline">X</span> satisfying <span class="latex-inline">AX = B</span>.
     */
    @Override
    T solve(T A, T B);
}
