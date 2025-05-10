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

package org.flag4j.arrays.backend.field_arrays;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.linalg.ops.common.ring_ops.RingProperties;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldElemDiv;
import org.flag4j.numbers.Field;

// TODO: Javadoc.
public abstract class AbstractDenseFieldMatrix<T extends AbstractDenseFieldMatrix<T, U, V>,
        U extends AbstractDenseFieldVector<U, T, V>, V extends Field<V>>
        extends AbstractDenseRingMatrix<T, U, V>
        implements MatrixMixin<T, T, U, V>, FieldTensorMixin<T, T, V> {


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param data Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    protected AbstractDenseFieldMatrix(Shape shape, V[] data) {
        super(shape, data);
    }


    /**
     * Constructs a sparse CSR matrix which is of a similar type as this dense matrix.
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    public abstract AbstractCsrFieldMatrix<?, T, ?, V> makeLikeCsrMatrix(
            Shape shape, V[] entries, int[] rowPointers, int[] colIndices);


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is approximately equal to its transpose.
     *
     * @return {@code true} if this matrix it is unitary; {@code false} otherwise.
     */
    public boolean isUnitary() {
        return numRows == numCols && mult(H()).isI();
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Matrix abs() {
        double[] abs = new double[data.length];
        RingOps.abs(data, abs);
        return new Matrix(shape, abs);
    }


    /**
     * Computes the Hermitian transpose of this matrix.
     *
     * @return The Hermitian transpose of this matrix.
     */
    @Override
    public T H() {
        V[] dest = makeEmptyDataArray(data.length);
        TransposeDispatcher.dispatchHermitian(data, shape, dest);
        return makeLikeNDArray(shape.swapAxes(0, 1), dest);
    }


    /**
     * Computes the element-wise quotient of two matrices.
     *
     * @param b Second matrix in the element-wise quotient.
     *
     * @return The element-wise quotient of this matrix and {@code b}.
     */
    @Override
    public T div(T b) {
        V[] dest = makeEmptyDataArray(data.length);
        DenseFieldElemDiv.dispatch(data, shape, b.data, b.shape, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Computes the element-wise square root of this tensor.
     *
     * @return The element-wise square root of this tensor.
     */
    @Override
    public T sqrt() {
        V[] dest = makeEmptyDataArray(data.length);
        FieldOps.sqrt(data, dest);
        return makeLikeNDArray(shape, dest);
    }


    /**
     * Checks if this tensor only contains finite values.
     *
     * @return {@code true} if this tensor only contains finite values; {@code false} otherwise.
     *
     * @see #containsInf()
     * @see #containsNaN()
     */
    @Override
    public boolean isAllFinite() {
        return FieldOps.isFinite(data);
    }


    /**
     * Checks if this tensor contains at least one infinite value.
     *
     * @return {@code true} if this tensor contains at least one infinite value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsNaN()
     */
    @Override
    public boolean containsInf() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if this tensor contains at least one NaN value.
     *
     * @return {@code true} if this tensor contains at least one NaN value; {@code false} otherwise.
     *
     * @see #isAllFinite()
     * @see #containsInf()
     */
    @Override
    public boolean containsNaN() {
        return FieldOps.isInfinite(data);
    }


    /**
     * Checks if all data of this matrix are "close" as defined below. Custom tolerances may be specified using
     * {@link #allClose(AbstractDenseFieldMatrix, double, double)}.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same shape and all data are "close" element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (1E-08 + 1E-05*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldMatrix, double, double) (AbstractDenseFieldTensor, double, double)
     */
    public boolean allClose(T b) {
        return hasSameShape(b) && RingProperties.allClose(data, b.data);
    }


    /**
     * Checks if all data of this matrix are "close" as defined below.
     * @param b Second tensor in the comparison.
     * @return True if both tensors have the same length and all data are "close" element-wise, i.e.
     * elements {@code x} and {@code y} at the same positions in the two tensors respectively and satisfy
     * {@code |x-y| <= (absTol + relTol*|y|)}. Otherwise, returns false.
     * @see #allClose(AbstractDenseFieldMatrix)
     */
    public boolean allClose(T b, double relTol, double absTol) {
        return hasSameShape(b) && RingProperties.allClose(data, b.data, relTol, absTol);
    }
}
