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
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.sparse.CooSemiringMatrix;
import org.flag4j.arrays.sparse.CooSemiringTensor;
import org.flag4j.arrays.sparse.CsrSemiringMatrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.numbers.BoolSemiring;
import org.flag4j.numbers.Field;
import org.flag4j.numbers.Semiring;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * <p>Instances of this class represents a dense matrix backed by a {@link Semiring} array. The {@code SemiringMatrix} class
 * provides functionality for matrix operations whose elements are members of a semiring, supporting mutable data with a fixed shape.
 *
 * <p>A {@code SemiringMatrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
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
 * <p>Using {@link BoolSemiring a boolean semiring}:
 * <pre>{@code
 * // Constructing an integer matrix from a 2D array of booleans
 * BoolSemiring[][] data = {
 *     { new BoolSemiring(true), new BoolSemiring(false) },
 *     { new BoolSemiring(true), new BoolSemiring(true) }
 * };
 * SemiringMatrix<BoolSemiring> matrix = new FieldMatrix(data);
 *
 * // Add matrices (equivalent to element-wise OR).
 * SemiringMatrix<BoolSemiring> sum = matrix.add(matrix);
 *
 * // Element-wise product of matrices (equivalent to element-wise AND).
 * SemiringMatrix<BoolSemiring> prod = matrix.elemMult(matrix);
 *
 * // Performing matrix multiplication.
 * SemiringMatrix<BoolSemiring> result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * SemiringMatrix<BoolSemiring> transpose = matrix.T();
 * }</pre>
 *
 * <p>Using {@link org.flag4j.numbers.Complex128 128-bit complex numbers}:
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers
 * Complex128[][] complexData = {
 *     { new Complex128(1, 2), new Complex128(3, 4) },
 *     { new Complex128(5, 6), new Complex128(7, 8) }
 * };
 * SemiringMatrix<Complex128> matrix = new FieldMatrix(complexData);
 *
 * // Performing matrix multiplication.
 * SemiringMatrix<Complex128> result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * SemiringMatrix<Complex128> transpose = matrix.T();
 * }</pre>
 *
 * @param <T> Type of the {@link Semiring semiring} element for the matrix.
 *
 * @see SemiringMatrix
 * @see SemiringVector
 * @see SemiringTensor
 * @see AbstractDenseSemiringMatrix
 */
public class SemiringMatrix<T extends Semiring<T>> extends AbstractDenseSemiringMatrix<
        SemiringMatrix<T>, SemiringVector<T>, T> {

    private static final long serialVersionUID = 1L;


    /**
     * Creates a tensor with the specified data and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all data within the tensor.
     * If this tensor is sparse, this specifies only the non-zero data of the tensor.
     */
    public SemiringMatrix(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense semiring matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public SemiringMatrix(int rows, int cos, T[] entries) {
        super(new Shape(rows, cos), entries);
    }


    /**
     * Creates a dense semiring matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public SemiringMatrix(T[][] entries) {
        super(new Shape(entries.length, entries[0].length), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense semiring matrix with the specified data and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public SemiringMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Semiring[shape.totalEntriesIntValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a dense semiring matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public SemiringMatrix(int rows, int cols, T[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense semiring matrix with the specified data and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public SemiringMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Semiring[rows*cols]);
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
    protected SemiringVector<T> makeLikeVector(Shape shape, T[] entries) {
        return new SemiringVector<>(shape, entries);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected SemiringVector<T> makeLikeVector(T[] entries) {
        return new SemiringVector<>(entries);
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
    protected CooSemiringMatrix<T> makeLikeCooMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooSemiringMatrix<T>(shape, entries, rowIndices, colIndices);
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
    public CsrSemiringMatrix<T> makeLikeCsrMatrix(
            Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrSemiringMatrix<T>(shape, entries, rowPointers, colIndices);
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
    protected CooSemiringTensor<T> makeLikeCooTensor(Shape shape, T[] data, int[][] indices) {
        return new CooSemiringTensor<>(shape, data, indices);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the {@code shape} and
     * {@code data}. The resulting tensor will also have
     * the same non-zero indices as this tensor.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type and with the same non-zero indices as this tensor with the given the {@code shape} and
     * {@code data}.
     */
    @Override
    public SemiringMatrix<T> makeLikeNDArray(Shape shape, T[] data) {
        return new SemiringMatrix<>(shape, data);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public SemiringTensor<T> toTensor() {
        return new SemiringTensor(shape, data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must have the same number of total entries as the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public SemiringTensor<T> toTensor(Shape newShape) {
        // The constructor should ensure that newShape.totalEntriesIntValueExact() == data.length.
        return new SemiringTensor(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of the specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape, Semiring)
     * @see #I(int, int, Semiring)
     */
    public static <T extends Semiring<T>> SemiringMatrix<T> I(int size, T fieldValue) {
        return I(size, size, fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of the specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Semiring)
     * @see #I(Shape, Semiring)
     */
    public static <T extends Semiring<T>> SemiringMatrix<T> I(int numRows, int numCols, T fieldValue) {
        return I(new Shape(numRows, numCols), fieldValue);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape The shape of the identity-like matrix to construct.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of the specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int, Semiring)
     * @see #I(Shape, Semiring)
     */
    public static <T extends Semiring<T>> SemiringMatrix<T> I(Shape shape, T fieldValue) {
        Field[] identityValues = new Field[shape.totalEntriesIntValueExact()];
        Arrays.fill(identityValues, (Field) fieldValue.getZero());
        Field one = (Field) fieldValue.getOne();

        int rows = shape.getSize(0);
        int cols = shape.getSize(1);

        for(int i=0, stop=Math.min(rows, cols); i<stop; i++)
            identityValues[i*cols + i] = one;

        return new SemiringMatrix(shape, identityValues);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern it is better to underestimate the sparsity.
     *
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     *
     * @see #toCoo()
     */
    @Override
    public CooSemiringMatrix<T> toCoo(double estimatedSparsity) {
        return (CooSemiringMatrix<T>) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooSemiringMatrix<T> toCoo() {
        return (CooSemiringMatrix<T>) super.toCoo();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr(double)
     */
    @Override
    public CsrSemiringMatrix<T> toCsr() {
        return (CsrSemiringMatrix<T>) super.toCsr();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @param estimatedSparsity Estimated sparsity of the matrix. Must be between 0 and 1 inclusive. If this is an accurate estimation
     * it <em>may</em> provide a slight speedup and can reduce unneeded memory consumption. If memory is a concern, it is better to
     * overestimate the sparsity. If speed is the concern it is better to underestimate the sparsity.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr()
     */
    @Override
    public CsrSemiringMatrix<T> toCsr(double estimatedSparsity) {
        return (CsrSemiringMatrix<T>) super.toCsr(estimatedSparsity);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <em>may</em> significantly more efficient than calling
     * {@code #mult(Matrix) this.mult(this)} {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public SemiringMatrix<T> pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return I(numRows, data[0]);
        if (n == 1) return copy();
        if (n == 2) return this.mult(this);

        SemiringMatrix<T> result = I(numRows, data[0]);  // Start with identity matrix.
        SemiringMatrix<T> base = this;

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
     * <p>This method will throw an {@code UnsupportedOperationException} as subtraction is not defined for a general semiring.
     */
    @Override
    public SemiringMatrix<T> sub(SemiringMatrix<T> b) {
        throw new UnsupportedOperationException("Cannot compute subtraction with matrix type: " + this.getClass().getName());
    }


    /**
     * <p>{@inheritDoc}
     * <p>This method will throw an {@code UnsupportedOperationException} as division is not defined for a general semiring.
     */
    @Override
    public SemiringMatrix<T> div(SemiringMatrix<T> b) {
        throw new UnsupportedOperationException("Cannot compute division with matrix type: " + this.getClass().getName());
    }


    /**
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type
     * {@link SemiringMatrix}; {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        SemiringMatrix<T> src2 = (SemiringMatrix<T>) object;

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
