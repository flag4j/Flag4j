/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.decompositions.balance;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.VectorNorms;
import org.flag4j.linalg.ops.common.real.RealOps;
import org.flag4j.linalg.ops.common.real.RealProperties;
import org.flag4j.linalg.ops.dense.real.RealDenseOps;
import org.flag4j.util.ValidateParameters;

/**
 * <p>Instances of this class may be used to balance real dense matrices. Balancing a matrix involves computing a
 * diagonal similarity transformation to "balance" the rows and columns of the matrix. This balancing is achieved
 * by attempting to scale the entries of the matrix by similarity transformations such that the 1-norms of corresponding
 * rows and columns have the similar 1-norms. Rows and columns may also be permuted during balancing if requested.
 *
 * <p>Balancing is often used as a preprocessing step to improve the conditioning of eigenvalue problems. Because the
 * balancing transformation is a similarity transformation, the eigenvalues are preserved. Further, when permutations are
 * done during balancing it is possible to isolate decoupled eigenvalues.
 *
 * <p>The similarity transformation of a square matrix <span class="latex-inline">A</span> into the
 * balanced matrix <span class="latex-inline">B</span> can be described as:
 * <span class="latex-eq-align">
 * <pre>
 *     B = T<sup>-1</sup> A T
 *       = D<sup>-1</sup> P<sup>-1</sup> A P D.</pre>
 * </span>
 *
 * Solving for <span class="latex-inline">A</span>,
 * balancing may be viewed as the following decomposition:
 * <span class="latex-eq-align">
 * <pre>
 *     A = T B T<sup>-1</sup>
 *       = P D B D<sup>-1</sup> P<sup>-1</sup>.</pre>
 * </span>
 *
 * Where <span class="latex-inline">P</span> is a permutation matrix,
 * and <span class="latex-inline">D</span> is a diagonal scaling matrix.
 *
 * <p>When permutations are used during balancing we obtain a specific form. First,
 * <span class="latex-replace">
 * <pre>
 *           <sup>  </sup>[ T<sub>1</sub>  X   Y  ]
 *   P<sup>-1</sup> A P = [  0  B<sub>1</sub>  Z  ]
 *           <sup>  </sup>[  0  0   T<sub>2</sub> ]</pre>
 * </span>
 *
 * <!-- LATEX:
 * {@literal
 * \[ \begin{align*}
 * P^{-1}AP = \begin{bmatrix}
 * T_1 & X & Y \\
 * \mathbf{0} & B_1 & Z \\
 * \mathbf{0} & \mathbf{0} & T_2
 * \end{bmatrix}
 * \end{align*} \]
 * }
 * -->
 *
 * Where <span class="latex-inline">T<sub>1</sub></span>
 * and <span class="latex-inline">T<sub>2</sub></span> are upper triangular
 * matrices whose eigenvalues lie along the diagonal.
 * These are also eigenvalues of <span class="latex-inline">A</span>.
 * Then, if scaling is applied we obtain:
 *
 * <span class="latex-replace">
 * <pre>
 *               <sup>    </sup>[ T<sub>1</sub>     X*D<sub>1</sub>       Y   ]
 *   D<sup>-1</sup> P<sup>-1</sup> A P D = [  0  D<sub>1</sub><sup>-1</sup>*B<sub>1</sub>*D<sub>1</sub>  D<sub>1</sub><sup>-1</sup>*Z  ]
 *               <sup>    </sup>[  0      0         T<sub>2</sub>  ]</pre>
 * </span>
 *
 * <!-- LATEX:
 * {@literal
 * \[ D^{-1}P^{-1}APD = \begin{bmatrix}
 * T_1 & XD_1 & Y \\
 * \mathbf{0} & D_1^{-1}B_1D_1 & D_1^{-1}Z \\
 * \mathbf{0} & \mathbf{0} & T_2
 * \end{bmatrix} \]
 * }
 * -->
 *
 * where <span class="latex-inline">D<sub>1</sub></span> is a diagonal matrix such that,
 * <span class="latex-replace">
 * <pre>
 *         [ I<sub>1</sub> 0  0  ]
 *     D = [ 0  D<sub>1</sub> 0  ]
 *         [ 0  0  I<sub>2</sub> ]</pre>
 * </span>
 *
 * <!-- LATEX:
 * {@literal
 * \[ D = \begin{bmatrix}
 * I_1 & \mathbf{0} & \mathbf{0} \\
 * \mathbf{0} & D_1 & \mathbf{0} \\
 * \mathbf{0} & \mathbf{0} & I_2
 * \end{bmatrix} \]
 * }
 * -->
 *
 * Where <span class="latex-inline">I<sub>1</sub></span> and
 * <span class="latex-inline">I<sub>2</sub></span> are identity matrices with
 * equivalent shapes to <span class="latex-inline">T<sub>1</sub></span> and
 * <span class="latex-inline">T<sub>2</sub></span>.
 *
 * <p>Once balancing has been applied, one need only compute the eigenvalues of
 * <span class="latex-inline">B<sub>1</sub></span> and combine them with the diagonal
 * entries of <span class="latex-inline">T<sub>1</sub></span> and
 * <span class="latex-inline">T<sub>2</sub></span>
 * to obtain all eigenvalues of <span class="latex-inline">A</span>.
 *
 * <p>The code in this class if heavily based on LAPACK's reference implementations of
 * <a href=https://www.netlib.org/lapack/explore-html/df/df3/group__gebal.html>xGEBAL</a> (v 3.12.1).
 *
 * @see #getB()
 * @see #getBSubMatrix()
 * @see #getD(boolean)
 * @see #getD()
 * @see #getP()
 * @see #getT()
 */
public class RealBalancer extends Balancer<Matrix> {


    /**
     * <p>Constructs a real balancer which will perform both the permutations and scaling steps out-of-place.
     *
     * <p>To specify if permutations or scaling should be or should not be performed, use {@link #RealBalancer(boolean, boolean)}.
     * To specify if the balancing should be done in-place, use {@link #RealBalancer(boolean, boolean, boolean)}.
     */
    public RealBalancer() {
        super(true, true, false);
    }


    /**
     * <p>Constructs a real balancer optionally performing the permutation and scaling steps out-of-place.
     *
     * <p>To specify if the balancing should be done in-place, use {@link #RealBalancer(boolean, boolean, boolean)}.
     *
     * @param doPermutations Flag indicating if the permutation step should be performed during balancing.
     * <ul>
     *     <li>If {@code true}: the permutation step will be performed.</li>
     *     <li>If {@code false}: the permutation step will <i>not</i> be performed.</li>
     * </ul>
     * @param doScaling Flag indicating if the scaling step should be performed during balancing.
     * <ul>
     *     <li>If {@code true}: the scaling step will be performed.</li>
     *     <li>If {@code false}: the scaling step will <i>not</i> be performed.</li>
     * </ul>
     */
    public RealBalancer(boolean doPermutations, boolean doScaling) {
        super(doPermutations, doScaling, false);
    }


    /**
     * <p>Constructs a real balancer optionally performing the permutation and scaling steps in/out-of-place.
     *
     * @param doPermutations Flag indicating if the permutation step should be performed during balancing.
     * <ul>
     *     <li>If {@code true}: the permutation step will be performed.</li>
     *     <li>If {@code false}: the permutation step will <i>not</i> be performed.</li>
     * </ul>
     * @param doScaling Flag indicating if the scaling step should be performed during balancing.
     * <ul>
     *     <li>If {@code true}: the scaling step will be performed.</li>
     *     <li>If {@code false}: the scaling step will <i>not</i> be performed.</li>
     * </ul>
     * @param inPlace Flag indicating if the balancing should be done in or out-of-place.
     * <ul>
     *     <li>If {@code true}: balancing will be done in-place and the source matrix will be overwritten.</li>
     *     <li>If {@code false}: balancing will be done out-of-place.</li>
     * </ul>
     */
    public RealBalancer(boolean doPermutations, boolean doScaling, boolean inPlace) {
        super(doScaling, doScaling, inPlace);
    }


    /**
     * Swaps two rows, over a specified range, within the {@link #balancedMatrix} matrix.
     *
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @param start Index of the column specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the column specifying the end of the range for the row swap (exclusive).
     */
    @Override
    protected void swapRows(int rowIdx1, int rowIdx2, int start, int stop) {
        RealDenseOps.swapRowsUnsafe(balancedMatrix.shape, balancedMatrix.data, rowIdx1, rowIdx2, start, stop);
    }


    /**
     * Swaps two columns, over a specified range, within the {@link #balancedMatrix} matrix.
     *
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @param start Index of the row specifying the start of the range for the column swap (inclusive).
     * @param stop Index of the row specifying the end of the range for the column swap (exclusive).
     */
    @Override
    protected void swapCols(int colIdx1, int colIdx2, int start, int stop) {
        RealDenseOps.swapColsUnsafe(balancedMatrix.shape, balancedMatrix.data, colIdx1, colIdx2, start, stop);
    }


    /**
     * Checks if a value within {@link #balancedMatrix} is zero.
     *
     * @param idx Index of value within flat data {@link #balancedMatrix} to check if it is zero.
     */
    @Override
    protected boolean isZero(int idx) {
        return balancedMatrix.data[idx] == 0.0;
    }


    /**
     * Computes the <span class="latex-inline">&ell;<sup>2</sup></span> norm of a vector with {@code n} 
     * elements from {@link #balancedMatrix}s 1D data array
     * starting at index {@code start} and spaced by {@code stride}.
     *
     * @param start Starting index within {@link #balancedMatrix}s 1D data array to compute norm of.
     * @param n The number of elements in the vector to compute norm of.
     * @param stride The spacing between each element within {@link #balancedMatrix}s 1D data array to norm of.
     *
     * @return The norm of the vector containing the specified elements from {@link #balancedMatrix}s 1D data array.
     */
    @Override
    protected double vectorNorm(int start, int n, int stride) {
        return VectorNorms.norm(balancedMatrix.data, start, n, stride);
    }


    /**
     * Computes the maximum absolute value of a vector with {@code n} elements from {@link #balancedMatrix}s 1D data array
     * starting at index {@code start} and spaced by {@code stride}.
     *
     * @param start Starting index within {@link #balancedMatrix}s 1D data array to compute maximum absolute value of.
     * @param n The number of elements in the vector to compute maximum absolute value of.
     * @param stride The spacing between each element within {@link #balancedMatrix}s 1D data array to compute maximum absolute
     * value of.
     *
     * @return The maximum absolute value of the vector containing the specified elements from {@link #balancedMatrix}s 1D data
     * array.
     */
    @Override
    protected double vectorMaxAbs(int start, int n, int stride) {
        return RealProperties.maxAbs(start, n, stride);
    }


    /**
     * Scales a vector with {@code n} elements from {@link #balancedMatrix}s 1D data array
     * starting at index {@code start} and spaced by {@code stride}. This operation must be done in-place.
     *
     * @param start Starting index within {@link #balancedMatrix}s 1D data array begin scaling.
     * @param n The number of elements to scale.
     * @param stride The spacing between each element within {@link #balancedMatrix}s 1D data array to scale.
     */
    @Override
    protected void vectorScale(double factor, int start, int n, int stride) {
        RealOps.scalMult(balancedMatrix.data, factor, start, n, stride, balancedMatrix.data);
    }


    /**
     * Efficiently left multiplies <span class="latex-inline">PD</span> to the provided {@code src} matrix.
     * @param src Matrix to apply transform to.
     * @return The result of left multiplying <span class="latex-inline">PD</span> to the {@code src} matrix.
     */
    @Override
    public Matrix applyLeftTransform(Matrix src) {
        ensureHasBalanced();
        ValidateParameters.ensureSquareMatrix(src.shape);
        ValidateParameters.ensureEqualShape(balancedMatrix.shape, src.shape);

        double[] destData = new double[src.data.length];
        System.arraycopy(src.data, 0, destData, 0, iLow*size);
        System.arraycopy(src.data, iHigh*size, destData, iHigh*size, (size - iHigh)*size);

        // Left multiply by D.
        for(int i = iLow; i < iHigh; i++) {
            int rowOffset = i*size;
            double scale = scalePerm[i];

            for(int j = 0; j < size; j++)
                destData[rowOffset + j] = scale*src.data[rowOffset + j];
        }

        return getP().leftMult(new Matrix(balancedMatrix.shape, destData));
    }


    /**
     * Efficiently right multiplies <span class="latex-inline">D<sup>-1</sup>P<sup>-1</sup></span> to the provided {@code src} matrix.
     * @param src Matrix to apply transform to.
     * @return The result of right multiplying <span class="latex-inline">D<sup>-1</sup>P<sup>-1</sup></span> to the {@code src} matrix.
     */
    @Override
    public Matrix applyRightTransform(Matrix src) {
        ensureHasBalanced();
        ValidateParameters.ensureSquareMatrix(src.shape);
        ValidateParameters.ensureEqualShape(balancedMatrix.shape, src.shape);

        double[] destData = new double[src.data.length];
        System.arraycopy(src.data, 0, destData, 0, iLow*size);
        System.arraycopy(src.data, iHigh*size, destData, iHigh*size, (size - iHigh)*size);

        // Right multiply by D^{-1}.
        for(int i = iLow; i < iHigh; i++) {
            int rowOffset = i*size;
            double scale = 1.0/scalePerm[i];

            for(int j = 0; j < size; j++)
                destData[rowOffset + j] = scale*src.data[rowOffset + j];
        }

        // Left multiply by P^{-1}.
        return getP().inv().rightMult(new Matrix(balancedMatrix.shape, destData));
    }
}
