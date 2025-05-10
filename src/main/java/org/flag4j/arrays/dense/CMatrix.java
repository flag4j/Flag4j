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
import org.flag4j.arrays.sparse.*;
import org.flag4j.io.PrettyPrint;
import org.flag4j.linalg.MatrixNorms;
import org.flag4j.linalg.ops.MatrixMultiplyDispatcher;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.linalg.ops.dense_sparse.csr.semiring_ops.DenseCsrSemiringMatMult;
import org.flag4j.linalg.ops.dispatch.Cm128DeMatMultDispatcher;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.Arrays;

/**
 * <p>Instances of this class represent a complex dense matrix backed by a {@link Complex128} array.
 * The {@code CMatrix} class provides functionality for complex matrix operations, supporting mutable data with a fixed shape.
 * This class extends {@link AbstractDenseFieldMatrix} and offers additional methods optimized for complex
 * arithmetic and matrix computations.
 *
 * <p>A {@code CMatrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank-n tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Construction from various data types such as arrays of {@link Complex128}, {@code double}, and {@link String}.</li>
 *   <li>Support for standard matrix operations like addition, subtraction, multiplication, and exponentiation.</li>
 *   <li>Conversion methods to other matrix representations, such as COO (Coordinate) and CSR (Compressed Sparse Row) formats.</li>
 *   <li>Utility methods for checking properties like being unitary, real, or complex.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers
 * Complex128[][] complexData = {
 *     { new Complex128(1, 2), new Complex128(3, 4) },
 *     { new Complex128(5, 6), new Complex128(7, 8) }
 * };
 * CMatrix matrix = new CMatrix(complexData);
 *
 * // Performing matrix multiplication.
 * CMatrix result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * CMatrix transpose = matrix.T();
 *
 * // Performing matrix conjugate transpose (i.e., Hermitian transpose).
 * CMatrix conjugateTranspose = matrix.H();
 *
 * // Checking if the matrix is unitary.
 * boolean isUnitary = matrix.isUnitary();
 * }</pre>
 *
 * @see Complex128
 * @see CVector
 * @see CTensor
 * @see AbstractDenseFieldMatrix
 */
public class CMatrix extends AbstractDenseFieldMatrix<CMatrix, CVector, Complex128> {
    private static final long serialVersionUID = 1L;


    /**
     * Creates a complex matrix with the specified {@code data} and {@code shape}.
     *
     * @param shape Shape of this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(Shape shape, Complex128[] entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code shape} filled with {@code fillValue}.
     *
     * @param shape Shape of this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(Shape shape, Complex128 fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a zero matrix with the specified {@code shape}.
     *
     * @param shape Shape of this matrix.
     */
    public CMatrix(Shape shape) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Creates a square zero matrix with the specified {@code size}.
     *
     * @param size Size of the zero matrix to construct. The resulting matrix will have shape {@code (size, size)}
     */
    public CMatrix(int size) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified {@code data}, and shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param entries Entries of this matrix.
     */
    public CMatrix(int rows, int cols, Complex128[] entries) {
        super(new Shape(rows, cols), entries);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a complex matrix with the specified shape and filled with {@code fillValue}.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     * @param fillValue Value to fill this matrix with.
     */
    public CMatrix(int rows, int cols, Complex128 fillValue) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a zero matrix with the specified shape.
     *
     * @param rows The number of rows in this matrix.
     * @param cols The number of columns in this matrix.
     */
    public CMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[rows*cols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, Complex128.ZERO);
    }


    /**
     * Constructs a complex matrix from a 2D array. The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(Complex128[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(Complex128[] row : entries) {
            for(Complex128 value : row)
                super.data[flatPos++] = value;
        }
    }


    /**
     * <p>Constructs a complex matrix from a 2D array of strings. Each string must be formatted properly as a complex number that can
     * be parsed by {@link org.flag4j.io.parsing.ComplexNumberParser}
     *
     * <p>The matrix will have the same shape as the array.
     * @param entries Entries of the matrix. Assumed to be a square array.
     */
    public CMatrix(String[][] entries) {
        super(new Shape(entries.length, entries[0].length), new Complex128[entries.length*entries[0].length]);
        setZeroElement(Complex128.ZERO);
        int flatPos = 0;

        for(String[] row : entries) {
            for(String value : row)
                super.data[flatPos++] = new Complex128(value);
        }
    }


    /**
     * Constructs a complex matrix with specified {@code shape} and {@code data}.
     * @param shape Shape of the matrix to construct.
     * @param entries Entries of the matrix.
     */
    public CMatrix(Shape shape, double[] entries) {
        super(shape, new Complex128[entries.length]);
        ValidateParameters.ensureRank(shape, 2);
        setZeroElement(Complex128.ZERO);
        ArrayUtils.arraycopy(entries, 0, super.data, 0, entries.length);
    }


    /**
     * Constructs a complex matrix from a 2D array of double values.
     * @param aEntriesReal Entries of the complex matrix to construct. Each value will be wrapped as {@link Complex128 Complex128's}.
     */
    public CMatrix(double[][] aEntriesReal) {
        super(new Shape(aEntriesReal.length, aEntriesReal[0].length), new Complex128[aEntriesReal.length*aEntriesReal[0].length]);
        setZeroElement(Complex128.ZERO);

        int idx = 0;
        for(double[] row : aEntriesReal) {
            for(double value : row)
                super.data[idx++] = new Complex128(value);
        }
    }


    /**
     * Constructs a matrix with the specified shape filled with {@code fillValue}.
     * @param numRows The number of rows in the matrix.
     * @param numCols The number of rows in the matrix.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int numRows, int numCols, double fillValue) {
        super(new Shape(numRows, numCols), new Complex128[numRows*numCols]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Complex128 fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, fillValue);
    }


    /**
     * Creates a square matrix with the specified {@code size} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(int size, Double fillValue) {
        super(new Shape(size, size), new Complex128[size*size]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Creates matrix with the specified {@code shape} filled with {@code fillValue}.
     * @param size Size of the square matrix to construct.
     * @param fillValue Value to fill matrix with.
     */
    public CMatrix(Shape shape, Double fillValue) {
        super(shape, new Complex128[shape.totalEntriesIntValueExact()]);
        setZeroElement(Complex128.ZERO);
        Arrays.fill(data, new Complex128(fillValue));
    }


    /**
     * Constructs a copy of the specified matrix.
     * @param mat Matrix to create a copy of.
     */
    public CMatrix(CMatrix mat) {
        super(mat.shape, mat.data.clone());
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
    protected CVector makeLikeVector(Shape shape, Complex128[] entries) {
        return new CVector(shape, entries);
    }


    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
    }


    /**
     * <p>Computes the Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm this matrix.
     *
     * <p>The Frobenius norm is defined as the square root of the sum of absolute squares of all entries in the matrix.
     *
     * @return the Frobenius of this tensor.
     */
    @Override
    public double norm() {
        return MatrixNorms.norm(this);
    }


    /**
     * <p>Computes the matrix operator norm of this matrix "induced" by the vector p-norm.
     * Specifically, this method computes the operator norm of the matrix as:
     * <span class="latex-replace"><pre>
     *     ||A||<sub>p</sub> = sup<sub>x&ne;0</sub>(||Ax||<sub>p</sub> / ||x||<sub>p</sub>).</pre></span>
     *
     * <!-- LATEX: \[ ||A||_p = \sup_{x \ne 0} \cfrac{||Ax||_p}{||x||_p} \] -->
     *
     * <p>This method supports a limited set of {@code p} values which yield simple formulas. When {@code p < 1}, the result this method
     * returns is not a true mathematical norm. However, these values may still be useful for numerical purposes.
     * <ul>
     *     <li>{@code p=1}: The maximum absolute column sum.</li>
     *     <li>{@code p=-1}: The minimum absolute column sum.</li>
     *     <li>{@code p=2}: The spectral norm. Equivalent to the largest singular value of the matrix.</li>
     *     <li>{@code p=-2}: The smallest singular value of the matrix.</li>
     *     <li>{@code p=Double.POSITIVE_INFINITY}: The maximum absolute row sum.</li>
     *     <li>{@code p=Double.NEGATIVE_INFINITY}: The minimum absolute row sum.</li>
     * </ul>
     *
     * @param p The p value in the "induced" p-norm. Must be one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     * @return Norm of the matrix.
     * @throws IllegalArgumentException If {@code p} is not one of the following: {@code 1}, {@code -1}, {@code 2}, {@code -2},
     * {@link Double#POSITIVE_INFINITY} or {@link Double#NEGATIVE_INFINITY}.
     */
    @Override
    public double norm(double p) {
        return MatrixNorms.inducedNorm(this, p);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of this matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm. Equivalent to the Euclidean norm of the vector containing the singular values of
     *     the matrix.</li>
     * </ul>
     *
     * <p>The <span class="latex-inline">L<sub>p,q</sub></span> norm is computed as if by:
     * <pre>{@code
     *      double norm = 0;
     *      for(int j=0; j<src.numCols; j++) {
     *          double sum = 0;
     *          for(int i=0; i<src.numRows; i++)
     *              sum += Math.pow(src.get(i, j).mag(), p);
     *
     *          norm += Math.pow(sum, q / p);
     *      }
     *
     *      return Math.pow(norm, 1.0 / q);
     * }</pre>
     *
     * @param p p value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @param q q value in the <span class="latex-inline">L<sub>p,q</sub></span> norm.
     * @return The <span class="latex-inline">L<sub>p,q</sub></span> norm of {@code src}.
     */
    public double norm(double p, double q) {
        return MatrixNorms.norm(this, p, q);
    }


    /**
     * Constructs a vector of a similar type as this matrix.
     *
     * @param entries Entries of the vector.
     *
     * @return A vector of a similar type as this matrix.
     */
    @Override
    protected CVector makeLikeVector(Complex128[] entries) {
        return new CVector(entries);
    }


    /**
     * Constructs a sparse COO matrix, which is of a similar type as this dense matrix.
     *
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     *
     * @return A sparse COO matrix which is of a similar type as this dense matrix.
     */
    @Override
    protected CooCMatrix makeLikeCooMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
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
    public CsrCMatrix makeLikeCsrMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols() != b.numRows()}.
     */
    @Override
    public CMatrix mult(CMatrix b) {
        return Cm128DeMatMultDispatcher.dispatch(this, b);
    }


    /**
     * Converts this matrix to an equivalent sparse COO matrix.
     *
     * @return A sparse COO matrix that is equivalent to this dense matrix.
     *
     * @see #toCoo(double)
     */
    @Override
    public CooCMatrix toCoo() {
        return (CooCMatrix) super.toCoo();
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
    public CooCMatrix toCoo(double estimatedSparsity) {
        return (CooCMatrix) super.toCoo(estimatedSparsity);
    }


    /**
     * Converts this matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix that is equivalent to this dense matrix.
     *
     * @see #toCsr(double)
     */
    @Override
    public CsrCMatrix toCsr() {
        return (CsrCMatrix) super.toCsr();
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
    public CsrCMatrix toCsr(double estimatedSparsity) {
        return (CsrCMatrix) super.toCsr(estimatedSparsity);
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
    protected CooCMatrix makeLikeCooTensor(Shape shape, Complex128[] entries, int[][] indices) {
        return makeLikeCooMatrix(shape, entries, indices[0], indices[1]);
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
    public CMatrix makeLikeNDArray(Shape shape, Complex128[] data) {
        return new CMatrix(shape, data);
    }


    /**
     * Constructs an empty matrix with the specified number of rows and columns. The data of the resulting matrix will be
     * all be {@code null}.
     * @param rows The number of rows in the matrix to construct.
     * @param cols The number of columns in the matrix to construct.
     * @return An empty matrix (i.e., filled with {@code null} values) with the specified shape.
     */
    public static CMatrix getEmpty(int rows, int cols) {
        return new CMatrix(rows, cols, new Complex128[rows*cols]);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(CooCMatrix b) {
        return (CMatrix) DenseCooFieldMatrixOps.add(this, b);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise sum between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise sum.
     *
     * @return The sum of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix add(CooMatrix b) {
        return (CMatrix) RealFieldDenseCooMatrixOps.add(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(CooCMatrix b) {
        return (CMatrix) DenseCooFieldMatrixOps.sub(this, b);
    }


    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }



    /**
     * Computes the element-wise difference between two tensors of the same shape.
     *
     * @param b Second tensor in the element-wise difference.
     *
     * @return The difference of this tensor with {@code b}.
     *
     * @throws ArrayShapeException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix sub(CooMatrix b) {
        return (CMatrix) RealFieldDenseCooMatrixOps.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CooCMatrix b) {
        Complex128[] dest = new Complex128[b.nnz];
        DenseCooFieldMatrixOps.elemMult(shape, data, b.shape, b.data, b.rowIndices, b.colIndices, dest);
        return CooCMatrix.unsafeMake(shape, dest, b.rowIndices.clone(), b.colIndices.clone());
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CMatrix elemMult(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemMult(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise multiplication of two tensors with the same shape.
     *
     * @param b Second tensor in the element-wise product.
     *
     * @return The element-wise product between this tensor and {@code b}.
     *
     * @throws IllegalArgumentException If this tensor and {@code b} do not have the same shape.
     */
    public CooCMatrix elemMult(CooMatrix b) {
        Complex128[] dest = new Complex128[b.nnz];
        RealFieldDenseCooMatrixOps.elemMult(this, b, dest);
        return CooCMatrix.unsafeMake(shape, dest, b.rowIndices.clone(), b.colIndices.clone());
    }


    /**
     * Computes the element-wise quotient between two tensors.
     *
     * @param b Second tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor with {@code b}.
     */
    public CMatrix div(Matrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.elemDiv(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor with the same shape and data as this matrix.
     */
    @Override
    public CTensor toTensor() {
        return new CTensor(shape, data.clone());
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified {@code newShape}.
     *
     * @param newShape Shape of the tensor. Can be any rank but must have the same number of total entries as the shape of this matrix.
     *
     * @return A tensor with the specified {@code newShape} and the same data as this matrix.
     */
    @Override
    public CTensor toTensor(Shape newShape) {
        ValidateParameters.ensureTotalEntriesEqual(shape, newShape);
        return new CTensor(newShape, data.clone());
    }


    /**
     * Constructs an identity matrix of the specified size.
     *
     * @param size Size of the identity matrix.
     * @return An identity matrix of the specified size.
     * @throws IllegalArgumentException If the specified size is less than 1.
     * @see #I(Shape)
     * @see #I(int, int)
     */
    public static CMatrix I(int size) {
        return I(size, size);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param numRows Number of rows in the identity-like matrix.
     * @param numCols Number of columns in the identity-like matrix.
     * @return An identity matrix of the specified shape.
     * @throws IllegalArgumentException If the specified number of rows or columns is less than 1.
     * @see #I(int)
     * @see #I(Shape)
     */
    public static CMatrix I(int numRows, int numCols) {
        ValidateParameters.ensureNonNegative(numRows, numCols);
        Complex128[] entries = new Complex128[numRows*numCols];
        Arrays.fill(entries, Complex128.ZERO);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++)
            entries[i*numCols+i] = Complex128.ONE;

        return new CMatrix(new Shape(numRows, numCols), entries);
    }


    /**
     * Constructs an identity-like matrix of the specified shape. That is, a matrix of zeros with ones along the
     * principle diagonal.
     *
     * @param shape Shape of the identity-like matrix.
     * @return An identity matrix of the specified size.
     * @throws IllegalArgumentException If the specified shape is not rank 2.
     * @see #I(int)
     * @see #I(int, int)
     */
    public static CMatrix I(Shape shape) {
        ValidateParameters.ensureRank(shape, 2);
        return I(shape.getSize(0), shape.getSize(1));
    }


    /**
     * Constructs a diagonal matrix from an array specifying the diagonal elements of the matrix.
     * @param data Diagonal elements of the matrix. All other values will be zero.
     * @return A diagonal matrix whose diagonal elements are equal to {@code data}.
     * @see #diag(CVector)
     */
    public static CMatrix diag(Complex128... data) {
        int size = data.length;
        Complex128[] fullData = new Complex128[size*size];
        Arrays.fill(fullData, Complex128.ZERO);

        int destIdx = 0;
        for(int i=0; i<size; i++) {
            fullData[destIdx] = data[i];
            destIdx += size + 1;
        }

        return new CMatrix(size, size, fullData);
    }


    /**
     * Constructs a diagonal matrix from a vector specifying the diagonal elements of the matrix.
     * @param vec Diagonal elements of the matrix. All other values will be zero.
     * @return A diagonal matrix whose diagonal elements are equal to the entries of {@code vec}.
     * @see #diag(Complex128...)
     */
    public static CMatrix diag(CVector vec) {
        return diag(vec.data);
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <em>may</em> be significantly more efficient than calling
     * {@link #mult(AbstractDenseSemiringMatrix) this.mult(this)} a total of {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     *
     * @throws IllegalArgumentException If this matrix is not square (i.e., {@code !this.isSquare()}).
     */
    public CMatrix pow(int n) {
        ValidateParameters.ensureSquare(shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return CMatrix.I(numRows);
        if (n == 1) return copy();
        if (n == 2) return mult(this);

        CMatrix result = CMatrix.I(numRows);  // Start with identity matrix.
        CMatrix base = this;

        // Compute the matrix power efficiently using an "exponentiation by squaring" approach.
        while(n > 0) {
            if((n & 1) == 1)  // If n is odd.
                result = result.mult(base);

            base = base.mult(base);  // Square the base.
            n >>= 1;  // Divide n by 2 (bitwise right shift).
        }

        return result;
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(Matrix b) {
        Complex128[] dest = MatrixMultiplyDispatcher.dispatch(this, b);
        return makeLikeNDArray(new Shape(numRows, b.numCols), dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(Vector b) {
        return MatrixMultiplyDispatcher.dispatch(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrMatrix b) {
        return (CMatrix) RealFieldDenseCsrMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CooMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows*b.numCols];
        RealFieldDenseCooMatMult.standard(
                data, shape, b.data, b.rowIndices, b.colIndices, b.shape, dest);
        Shape shape = new Shape(this.numRows, b.numCols);

        return new CMatrix(shape, dest);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrCMatrix b) {
        return (CMatrix) DenseCsrSemiringMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If {@code this.numCols() != b.numRows()}.
     */
    public CMatrix mult(CooCMatrix b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows*b.numCols];
        DenseCooFieldMatMult.standard(
                data, shape, b.data, b.rowIndices, b.colIndices, b.shape, dest);
        Shape shape = new Shape(numRows, b.numCols);

        return new CMatrix(shape, dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(CooVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows];
        RealFieldDenseCooMatMult.blockedVector(data, shape, b.data, b.indices, dest);
        return new CVector(dest);
    }


    /**
     * Computes the matrix-vector multiplication between this matrix and a vector.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the number
     *                                of length of the vector {@code b}.
     */
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[numRows];
        DenseCooFieldMatMult.blockedVector(data, shape, b.data, b.indices, dest);
        return new CVector(dest);
    }


    /**
     * Converts this complex matrix to a real matrix. This conversion is done by taking the real component of each entry and
     * ignoring the imaginary component.
     * @return A real matrix containing the real components of the data of this matrix.
     */
    public Matrix toReal() {
        return new Matrix(shape, Complex128Ops.toReal(data));
    }


    /**
     * Checks if all data of this matrix are real.
     * @return {@code true} if all data of this matrix are real; {@code false} otherwise.
     */
    public boolean isReal() {
        return Complex128Ops.isReal(data);
    }


    /**
     * Checks if any entry within this matrix has non-zero imaginary component.
     * @return {@code true} if any entry of this matrix has a non-zero imaginary component.
     */
    public boolean isComplex() {
        return Complex128Ops.isComplex(data);
    }


    /**
     * Checks if this matrix is unitary. That is, if the inverse of this matrix is approximately equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix it is unitary; {@code false} otherwise.
     */
    public boolean isUnitary() {
        return numRows == numCols && mult(H()).isCloseToIdentity();
    }


    /**
     * Rounds all data in this matrix to the nearest integer. The real and imaginary components will be rounded
     * independently.
     * @return A new matrix containing the data of this matrix rounded to the nearest integer.
     */
    public CMatrix round() {
        return round(0);
    }


    /**
     * Rounds all data within this matrix to the specified precision. The real and imaginary components will be rounded
     * independently.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new matrix containing the data of this matrix rounded to the specified precision.
     */
    public CMatrix round(int precision) {
        return new CMatrix(shape, Complex128Ops.round(data, precision));
    }


    /**
     * Sets all elements of this matrix to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this matrix with all data within {@code tol} of zero set to zero.
     */
    public CMatrix roundToZero(double tolerance) {
        return new CMatrix(shape, Complex128Ops.roundToZero(data, tolerance));
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
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type {@link CMatrix}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        CMatrix src2 = (CMatrix) object;

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
