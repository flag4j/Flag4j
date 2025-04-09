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
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.sparse.CooFieldMatrix;
import org.flag4j.arrays.sparse.CsrFieldMatrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.numbers.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;


/**
 * <p>Instances of this class represents a dense matrix backed by a {@link Field} array. The {@code FieldMatrix} class
 * provides functionality for matrix operations whose elements are members of a field, supporting mutable data with a fixed shape.
 *
 * <p>A {@code FieldMatrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank-n tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Support for standard matrix operations like addition, subtraction, multiplication, and exponentiation.</li>
 *   <li>Conversion methods to other matrix representations, such as COO (Coordinate) and CSR (Compressed Sparse Row) formats.</li>
 *   <li>Utility methods for checking properties like being triangular.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers (could be any field).
 * Complex128[][] complexData = {
 *     { new Complex128(1, 2), new Complex128(3, 4) },
 *     { new Complex128(5, 6), new Complex128(7, 8) }
 * };
 * FieldMatrix<Complex128> matrix = new FieldMatrix(complexData);
 *
 * // Performing matrix multiplication with the transpose of the matrix.
 * FieldMatrix<Complex128> result = matrix.mult(matrix.T());
 *
 * // Performing matrix conjugate transpose (i.e. Hermitian transpose).
 * FieldMatrix<Complex128> conjugateTranspose = matrix.H();  // May not be supported for all field types.
 *
 * // Checking if the matrix is upper triangular.
 * boolean isTriU = matrix.isTriU();
 * }</pre>
 *
 * @param <T> Type of the {@link Field field} for elements of the matrix.
 *
 * @see FieldVector
 * @see FieldTensor
 * @see AbstractDenseFieldMatrix
 */
public class FieldMatrix<T extends Field<T>> extends AbstractDenseFieldMatrix<FieldMatrix<T>, FieldVector<T>, T> {
    private static final long serialVersionUID = 1L;

    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T[] entries) {
        super(shape, entries);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cos, T[] entries) {
        super(new Shape(rows, cos), entries);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(T[][] entries) {
        super(new Shape(entries.length, entries[0].length), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified data and filled with {@code filledValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(Shape shape, T fillValue) {
        super(shape, (T[]) new Field[shape.totalEntriesIntValueExact()]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a dense field matrix with the specified data and shape.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T[][] entries) {
        super(new Shape(rows, cols), ArrayUtils.flatten(entries));
    }


    /**
     * Creates a dense field matrix with the specified data and filled with {@code filledValue}.
     *
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param fillValue Entries of this matrix.
     */
    public FieldMatrix(int rows, int cols, T fillValue) {
        super(new Shape(rows, cols), (T[]) new Field[rows*cols]);
        Arrays.fill(data, fillValue);
    }


    /**
     * Constructs a sparse COO tensor which is of a similar type as this dense tensor.
     *
     * @param shape Shape of the COO tensor.
     * @param entries Non-zero data of the COO tensor.
     * @param indices
     *
     * @return A sparse COO tensor which is of a similar type as this dense tensor.
     */
    @Override
    protected CooFieldMatrix<T> makeLikeCooTensor(Shape shape, T[] entries, int[][] indices) {
        return makeLikeCooMatrix(shape, entries, indices[0], indices[1]);
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
    protected FieldVector<T> makeLikeVector(Shape shape, T[] entries) {
        return new FieldVector<>(shape, entries);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given the shape and data.
     *
     * @param shape Shape of the tensor to construct.
     * @param entries Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given the shape and data.
     */
    @Override
    public FieldMatrix<T> makeLikeTensor(Shape shape, T[] entries) {
        return new FieldMatrix<T>(shape, entries);
    }


    /**
     * Constructs a vector of similar type to this matrix with the given {@code data}.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of similar type to this matrix with the given {@code data}.
     */
    @Override
    public FieldVector<T> makeLikeVector(T... entries) {
        return new FieldVector<T>(entries);
    }


    /**
     * Constructs a sparse CSR matrix of similar type to this dense matrix.
     *
     * @param shape Shape of the CSR matrix.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Row pointers of the CSR matrix.
     * @param colIndices Column indices of the non-zero data in the CSR matrix.
     *
     * @return A sparse CSR matrix with the specified shape and non-zero data.
     */
    @Override
    public CsrFieldMatrix<T> makeLikeCsrMatrix(Shape shape, T[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrFieldMatrix<T>(shape, entries, rowPointers, colIndices);
    }


    /**
     * Constructs a sparse COO matrix of similar type to this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Row indices of the non-zero data in the COO matrix.
     * @param colIndices Column indices of the non-zero data in the COO matrix.
     *
     * @return A sparse COO matrix with the specified shape and non-zero data.
     */
    @Override
    public CooFieldMatrix<T> makeLikeCooMatrix(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        return new CooFieldMatrix<T>(shape, entries, rowIndices, colIndices);
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
    public CooFieldMatrix<T> toCoo(double estimatedSparsity) {
        return (CooFieldMatrix<T>) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     */
    @Override
    public CooFieldMatrix<T> toCoo() {
        return (CooFieldMatrix<T>) super.toCoo();
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr(double)
     */
    @Override
    public CsrFieldMatrix<T> toCsr() {
        return (CsrFieldMatrix<T>) super.toCsr();
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
    public CsrFieldMatrix<T> toCsr(double estimatedSparsity) {
        return (CsrFieldMatrix<T>) super.toCsr(estimatedSparsity);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public FieldTensor<T> toTensor() {
        return new FieldTensor<>(new Shape(data.length), data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must have the same number of total entries as the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public FieldTensor<T> toTensor(Shape newShape) {
        return new FieldTensor<>(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @param fieldValue Value of field to create identity matrix for.
     * @return An identity matrix of specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape, Field)
     * @see #I(int, int, Field)
     */
    public static <T extends Field<T>> FieldMatrix<T> I(int size, T fieldValue) {
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
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static <T extends Field<T>> FieldMatrix<T> I(int numRows, int numCols, T fieldValue) {
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
     * @see #I(int, Field)
     * @see #I(Shape, Field)
     */
    public static <T extends Field<T>> FieldMatrix<T> I(Shape shape, T fieldValue) {
        Field[] identityValues = new Field[shape.totalEntriesIntValueExact()];
        Arrays.fill(identityValues, (Field) fieldValue.getZero());
        Field one = (Field) fieldValue.getOne();

        int rows = shape.get(0);
        int cols = shape.get(1);

        for(int i=0, stop=Math.min(rows, cols); i<stop; i++)
            identityValues[i*cols + i] = one;

        return new FieldMatrix(shape, identityValues); 
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <em>may</em> significantly more efficient than calling
     * {@link #mult(AbstractDenseSemiringMatrix) this.mult(this)} {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public FieldMatrix<T> pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return I(numRows, data[0]);
        if (n == 1) return copy();
        if (n == 2) return this.mult(this);

        FieldMatrix<T> result = I(numRows, data[0]);  // Start with identity matrix.
        FieldMatrix<T> base = this;

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
     * Checks if an object is equal to this matrix object.
     * @param object Object to check equality with this matrix.
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type
     * {@link FieldMatrix} {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        FieldMatrix<T> src2 = (FieldMatrix<T>) object;

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
