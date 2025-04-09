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

package org.flag4j.arrays.dense;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.ring_arrays.AbstractDenseRingMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.sparse.CooRingMatrix;
import org.flag4j.arrays.sparse.CooRingTensor;
import org.flag4j.arrays.sparse.CsrRingMatrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.linalg.ops.common.ring_ops.RingOps;
import org.flag4j.numbers.Field;
import org.flag4j.numbers.Ring;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * <p>Instances of this class represents a dense matrix backed by a {@link Ring} array. The {@code RingMatrix} class
 * provides functionality for matrix operations whose elements are members of a ring, supporting mutable data with a fixed shape.
 *
 * <p>A {@code RingMatrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank-n tensors.
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Support for standard matrix operations like addition, subtraction, multiplication, and exponentiation.</li>
 *   <li>Conversion methods to other matrix representations, such as COO (Coordinate) and CSR (Compressed Sparse Row) formats.</li>
 *   <li>Utility methods for checking properties like being unitary, real, or complex.</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <p>Using {@link org.flag4j.numbers.RealInt32 32-bit real integers}:
 * <pre>{@code
 * // Constructing an integer matrix from a 2D array.
 * RealInt32[][] data = {
 *     { new RealInt32(5), new RealInt32(-3) },
 *     { new RealInt32(-7), new RealInt32(8) }
 * };
 * RingMatrix<RealInt32> matrix = new FieldMatrix(data);
 *
 * // Performing matrix multiplication.
 * RingMatrix<RealInt32> result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * RingMatrix<RealInt32> transpose = matrix.T();
 * }</pre>
 *
 * <p>Using {@link org.flag4j.numbers.Complex128 128-bit complex number}:
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers
 * Complex128[][] data = {
 *     { new Complex128(1, 2), new Complex128(3, 4) },
 *     { new Complex128(5, 6), new Complex128(7, 8) }
 * };
 * RingMatrix<Complex128> matrix = new FieldMatrix(data);
 *
 * // Performing matrix multiplication.
 * RingMatrix<Complex128> result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * RingMatrix<Complex128> transpose = matrix.T();
 * }</pre>
 *
 * @param <T> Type of the {@link Ring ring} element for the matrix.
 *
 * @see RingMatrix
 * @see RingVector
 * @see RingTensor
 * @see AbstractDenseRingMatrix
 */
public class RingMatrix<T extends Ring<T>> extends AbstractDenseRingMatrix<
        RingMatrix<T>, RingVector<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    public RingMatrix(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense ring matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public RingMatrix(int rows, int cos, T[] entries) {
        super(new Shape(rows, cos), entries);
    }


    /**
     * Creates a dense ring matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public RingMatrix(T[][] entries) {
        super(new Shape(entries.length, entries[0].length), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense ring matrix with the specified data and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public RingMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Ring[shape.totalEntriesIntValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a dense ring matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public RingMatrix(int rows, int cols, T[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense ring matrix with the specified data and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public RingMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Ring[rows*cols]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param shape Shape of the vector to construct. Must be rank 1.
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected RingVector<T> makeLikeVector(Shape shape, T[] entries) {
        return new RingVector<>(shape, entries);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected RingVector<T> makeLikeVector(T[] entries) {
        return new RingVector<>(entries);
    }


    /**
     * Constructs a sparse COO matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     *
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    @Override
    protected CooRingMatrix<T> makeLikeCooMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooRingMatrix<T>(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse CSR matrix which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A sparse CSR matrix which is of a similar type as this dense matrix.
     */
    @Override
    public CsrRingMatrix<T> makeLikeCsrMatrix(
            Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrRingMatrix<T>(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param data Non-zero data of the COO tensor.
     * @param indices Non-zero indices of the COO tensor.
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooRingTensor<T> makeLikeCooTensor(Shape shape, T[] data, int[][] indices) {
        return new CooRingTensor<>(shape, data, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public RingMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new RingMatrix<>(shape, entries);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public RingTensor<T> toTensor() {
        return new RingTensor(shape, data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must have the same number of total entries as the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public RingTensor<T> toTensor(Shape newShape) {
        // The constructor should ensure that newShape.totalEntriesIntValueExact() == data.length.
        return new RingTensor(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape, Ring)
     * @see #I(int, int, Ring)
     */
    public static <T extends Ring<T>> RingMatrix<T> I(int size, T fieldValue) {
        return I(size, size, fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Ring)
     * @see #I(Shape, Ring)
     */
    public static <T extends Ring<T>> RingMatrix<T> I(int numRows, int numCols, T fieldValue) {
        return I(new Shape(numRows, numCols), fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape The shape of the identity-like matrix to construct.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Ring)
     * @see #I(Shape, Ring)
     */
    public static <T extends Ring<T>> RingMatrix<T> I(Shape shape, T fieldValue) {
        Field[] identityValues = new Field[shape.totalEntriesIntValueExact()];
        Arrays.fill(identityValues, (Field) fieldValue.getZero());
        Field one = (Field) fieldValue.getOne();

        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int i=0, stop=Math.min(rows, cols); i<stop; i++)
            identityValues[i*cols + i] = one;

        return new RingMatrix(shape, identityValues);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     *
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     *
     * @see #toCoo()
     */
    @Override
    public CooRingMatrix<T> toCoo(double estimatedSparsity) {
        return (CooRingMatrix<T>) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooRingMatrix<T> toCoo() {
        return (CooRingMatrix<T>) super.toCoo();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr(double)
     */
    @Override
    public CsrRingMatrix<T> toCsr() {
        return (CsrRingMatrix<T>) super.toCsr();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * over-estimate the sparsity. If speed is the concern it is better to under-estimate the sparsity.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr()
     */
    @Override
    public CsrRingMatrix<T> toCsr(double estimatedSparsity) {
        return (CsrRingMatrix<T>) super.toCsr(estimatedSparsity);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <em>may</em> significantly more efficient than calling
     * {@code #mult(Matrix) this.mult(this)} {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public RingMatrix<T> pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return I(numRows, data[0]);
        if (n == 1) return copy();
        if (n == 2) return this.mult(this);

        RingMatrix<T> result = I(numRows, data[0]);  // Start with identity matrix.
        RingMatrix<T> base = this;

        // Compute the matrix power efficiently using an "exponentiation by squaring" approach.
        while(n > 0) {
            // If n is odd.
            if((n & 1) == 1)  result = result.mult(base);

            base = base.mult(base);  // Square the base.
            n >>= 1;  // Divide n by 2 (bitwise right shift).
        }

        return result;
    }


    /**
     * Accepts a visitor that implements the {@link MatrixVisitor} interface.
     * This method is part of the "Visitor Pattern" and allows operations to be performed
     * on the matrix without modifying the matrix's class directly.
     *
     * @param visitor The visitor implementing the operation to be performed.
     *
     * @return The result of the visitor's operation, typically another matrix or a scalar value.
     *
     * @throws NullPointerException if the visitor is {@code null}.
     */
    @Override
    public <R> R accept(MatrixVisitor<R> visitor) {
        return visitor.visit(this);
    }


    /**
     * <p>{@inheritDoc}
     * <p>This method will throw an {@code UnsupportedOperationException} as subtraction is not defined for a general ring.
     */
    @Override
    public RingMatrix<T> sub(RingMatrix<T> b) {
        throw new UnsupportedOperationException("Cannot compute subtraction with matrix type: " + this.getClass().getName());
    }


    /**
     * Computes the element-wise absolute value of this tensor.
     *
     * @return The element-wise absolute value of this tensor.
     */
    @Override
    public Matrix abs() {
        double[] dest = new double[data.length];
        RingOps.abs(data, dest);
        return new Matrix(shape, dest);
    }


    /**
     * <p>{@inheritDoc}
     * <p>This method will throw an {@code UnsupportedOperationException} as division is not defined for a general ring.
     */
    @Override
    public RingMatrix<T> div(RingMatrix<T> b) {
        throw new UnsupportedOperationException("Cannot compute division with matrix type: " + this.getClass().getName());
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type
     * {@link RingMatrix}; {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        RingMatrix<T> src2 = (RingMatrix<T>) object;

        return shape.equals(src2.shape) && Arrays.equals(data, src2.data);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = 31*hash + shape.hashCode();
        hash = 31*hash + Arrays.hashCode(data);

        return hash;
    }


    /**
     * Generates a human-readable string representing this matrix.
     * @return A human-readable string representing this matrix.
     */
    @Override
    public String toString() {
        return PrettyPrint.matrixToString(shape, data);
    }
}
