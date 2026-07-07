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

package org.flag4j.linalg.solvers.exact;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.semiring_arrays.TensorOverSemiring;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.arrays.dense.Vector;

/**
 * <p>Solver for solving a real well determined linear tensor equation <span class="latex-inline">AX = B</span> in an exact sense.
 *
 * <p>All indices of <span class="latex-inline">X</span> are summed over in the tensor product with the rightmost indices of
 * <span class="latex-inline">A</span> as if by
 * {@link TensorOverSemiring#tensorDot(TensorOverSemiring, int[], int[]) A.tensorDot(X, M, N)} where
 * {@code M = new int[]{X.rank()-1, X.rank(), X.rank()+1, ..., A.rank()-1}} and
 * {@code N = new int[]{0, 1, ..., X.rank()-1}}.
 *
 * @see ComplexExactSolver
 * @see ExactTensorSolver
 * @see org.flag4j.linalg.TensorInvert
 */
public class RealExactTensorSolver extends ExactTensorSolver<Tensor, Matrix, Vector> {

    /**
     * Creates an exact tensor solver for solving a well determined linear tensor equation <span class="latex-inline">AX = B</span>
     * for <span class="latex-inline">X</span> in an exact sense.
     */
    public RealExactTensorSolver() {
        super(new RealExactSolver());
    }


    /**
     * Initializes matrix for equivalent linear matrix equation.
     *
     * @param A Tensor to convert to matrix.
     * @param prod Product of all axis lengths in {@code A}.
     * @return A matrix with the same items as tensor {@code A} with shape (prod, prod).
     */
    @Override
    protected Matrix initMatrix(Tensor A, int prod) {
        return new Matrix(prod, prod, A.data);
    }


    /**
     * Initializes vector for equivalent linear matrix equation.
     *
     * @param B Tensor to convert to vector.
     * @return Flattens tensor {@code B} and converts to a vector.
     */
    @Override
    protected Vector initVector(Tensor B) {
        return new Vector(B.data);
    }


    /**
     * Wraps solution as a tensor and reshapes to the proper shape.
     *
     * @param x Vector solution to linear matrix equation which is equivalent to the tensor equation
     * <span class="latex-inline">AX = B</span>.
     * @param outputShape Shape for the solution tensor <span class="latex-inline">X</span>.
     * @return The solution <span class="latex-inline">X</span> to the linear tensor equation <span class="latex-inline">AX = B</span>.
     */
    @Override
    protected Tensor wrap(Vector x, Shape outputShape) {
        return new Tensor(outputShape, x.data);
    }
}
