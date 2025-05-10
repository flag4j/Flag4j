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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.field_arrays.AbstractCooFieldMatrix;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.io.PrettyPrint;
import org.flag4j.io.PrintOptions;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.dense.real.RealDenseTranspose;
import org.flag4j.linalg.ops.dense_sparse.coo.field_ops.DenseCooFieldMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseCooMatOps;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexCooConcats;
import org.flag4j.linalg.ops.sparse.coo.real_complex.RealComplexSparseMatOps;
import org.flag4j.linalg.ops.sparse.coo.ring_ops.CooRingMatrixOps;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringEquals;
import org.flag4j.linalg.ops.sparse.coo.semiring_ops.CooSemiringMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.StringUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.ArrayShapeException;
import org.flag4j.util.exceptions.LinearAlgebraException;

import java.util.ArrayList;
import java.util.List;


/**
 * <p>A complex sparse matrix stored in coordinate list (COO) format. The {@link #data} of this COO tensor are
 * primitive doubles.
 *
 * <p>The {@link #data non-zero data} and non-zero indices of a COO matrix are mutable but the {@link #shape}
 * and total number of non-zero data is fixed.
 *
 * <p>COO matrices are well-suited for incremental matrix construction and modification but may not have ideal efficiency for matrix
 * ops like matrix multiplication. For heavy computations, it may be better to construct a matrix as a {@code CooMatrix} then
 * convert to a {@link CsrCMatrix} (using {@link #toCsr()}) as CSR (compressed sparse row) matrices are generally better suited for
 * efficient
 * matrix ops.
 *
 * <h2>COO Representation:</h2>
 * A sparse COO matrix is stored as:
 * <ul>
 *     <li>The full {@link #shape shape} of the matrix.</li>
 *     <li>The non-zero {@link #data} of the matrix. All other data in the matrix are
 *     assumed to be zero. Zero values can also explicitly be stored in {@link #data}.</li>
 *     <li>The {@link #rowIndices row indices} of the non-zero values in the sparse matrix.</li>
 *     <li>The {@link #colIndices column indices} of the non-zero values in the sparse matrix.</li>
 * </ul>
 *
 * <p>Note: many ops assume that the data of the COO matrix are sorted lexicographically by the row and column indices.
 * (i.e.) by row indices first then column indices. However, this is not explicitly verified. Any ops implemented in this
 * class will preserve the lexicographical sorting.
 *
 * <p>If indices need to be sorted, call {@link #sortIndices()}.
 */
public class CooCMatrix extends AbstractCooFieldMatrix<CooCMatrix, CMatrix, CooCVector, Complex128> {

    private static final long serialVersionUID = 1L;

    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, entries, rowIndices, colIndices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(shape,
                entries.toArray(new Complex128[entries.size()]),
                ArrayConversions.fromIntegerList(rowIndices),
                ArrayConversions.fromIntegerList(colIndices));
        if(entries.size() == 0 || entries.get(0) == null)
            setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param shape The shape of the matrix.
     */
    public CooCMatrix(Shape shape) {
        super(shape, new Complex128[0], new int[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(rows, cols), entries, rowIndices, colIndices);
        if(entries.length == 0 || entries[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int rows, int cols, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        super(new Shape(rows, cols),
                entries.toArray(new Complex128[entries.size()]),
                ArrayConversions.fromIntegerList(rowIndices),
                ArrayConversions.fromIntegerList(colIndices));
        if(super.data.length == 0 || super.data[0] == null) setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a zero matrix of the specified shape.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     */
    public CooCMatrix(int rows, int cols) {
        super(new Shape(rows, cols), new Complex128[0], new int[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Creates a sparse coo matrix with the specified non-zero data, non-zero indices, and shape.
     *
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @param entries Non-zero data of this sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(Shape shape, double[] entries, int[] rowIndices, int[] colIndices) {
        super(shape, ArrayConversions.toComplex128(entries, null), rowIndices, colIndices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a square sparse COO matrix with the specified {@code size} filled with zeros.
     * @param size Size of the square matrix to construct.
     */
    public CooCMatrix(int size) {
        super(new Shape(size, size), new Complex128[0], new int[0], new int[0]);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a square complex sparse COO matrix with the specified {@code size}, non-zero values, and non-zero indices.
     * @param size Size of the square matrix to construct.
     * @param entries Non-zero data of the sparse matrix.
     * @param rowIndices Non-zero row indices of this sparse matrix.
     * @param colIndices Non-zero column indies of this sparse matrix.
     */
    public CooCMatrix(int size, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        super(new Shape(size, size), entries, rowIndices, colIndices);
        setZeroElement(Complex128.ZERO);
    }


    /**
     * Constructs a copy of the specified complex sparse COO matrix.
     * @param b Matrix to copy.
     */
    public CooCMatrix(CooCMatrix b) {
        super(b.shape, b.data.clone(), b.rowIndices.clone(), b.colIndices.clone());
    }


    /**
     * Constructor useful for avoiding parameter validation while constructing COO matrices.
     * @param shape The shape of the matrix to construct.
     * @param data The non-zero data of this COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @param dummy Dummy object to distinguish this constructor from the safe variant. It is completely ignored in this constructor.
     */
    private CooCMatrix(Shape shape, Complex128[] data, int[] rowIndices, int[] colIndices, Object dummy) {
        // This constructor is hidden and called by unsafeMake to emphasize that creating a COO tensor in this manner is unsafe.
        super(shape, data, rowIndices, colIndices, dummy);
    }


    /**
     * <p>Factory to construct a COO matrix which bypasses any validation checks on the data and indices.
     * <p><strong>Warning:</strong> This method should be used with extreme caution. It primarily exists for internal use. Only use
     * this factory if you are 100% certain the parameters are valid as some methods may
     * throw exceptions or exhibit undefined behavior.
     * @param shape The full size of the COO matrix.
     * @param data The non-zero data of the COO matrix.
     * @param rowIndices The non-zero row indices of the COO matrix.
     * @param colIndices The non-zero column indices of the COO matrix.
     * @return A COO matrix constructed from the provided parameters.
     */
    public static CooCMatrix unsafeMake(Shape shape, Complex128[] data, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, data, rowIndices, colIndices, null);
    }



    @Override
    public Complex128[] makeEmptyDataArray(int length) {
        return new Complex128[length];
    }
    
    
    /**
     * Constructs a sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A sparse COO tensor of the same type as this tensor with the specified non-zero data and indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, Complex128[] entries, int[] rowIndices, int[] colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a COO matrix with the specified shape, non-zero data, and non-zero indices.
     *
     * @param shape Shape of the matrix.
     * @param entries Non-zero values of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     *
     * @return A COO matrix with the specified shape, non-zero data, and non-zero indices.
     */
    @Override
    public CooCMatrix makeLikeTensor(Shape shape, List<Complex128> entries, List<Integer> rowIndices, List<Integer> colIndices) {
        return new CooCMatrix(shape, entries, rowIndices, colIndices);
    }


    /**
     * Constructs a sparse COO vector of a similar type to this COO matrix.
     *
     * @param shape Shape of the vector. Must be rank 1.
     * @param entries Non-zero data of the COO vector.
     * @param indices Non-zero indices of the COO vector.
     *
     * @return A sparse COO vector of a similar type to this COO matrix.
     */
    @Override
    public CooCVector makeLikeVector(Shape shape, Complex128[] entries, int[] indices) {
        return new CooCVector(shape.totalEntriesIntValueExact(), entries, indices);
    }


    /**
     * Constructs a dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     *
     * @param shape Shape of the dense tensor.
     * @param entries Entries of the dense tensor.
     *
     * @return A dense tensor with the specified {@code shape} and {@code data} which is a similar type to this sparse tensor.
     */
    @Override
    public CMatrix makeLikeDenseTensor(Shape shape, Complex128[] entries) {
        return new CMatrix(shape, entries);
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
    public CooCMatrix makeLikeNDArray(Shape shape, Complex128[] data) {
        return new CooCMatrix(shape, data, rowIndices.clone(), colIndices.clone());
    }


    /**
     * Computes the tensor contraction of this tensor with a specified tensor over the specified set of axes. That is,
     * computes the sum of products between the two tensors along the specified set of axes.
     *
     * @param src2 Tensor to contract with this tensor.
     * @param aAxes Axes along which to compute products for this tensor.
     * @param bAxes Axes along which to compute products for {@code src2} tensor.
     *
     * @return The tensor dot product over the specified axes.
     *
     * @throws IllegalArgumentException If the two tensors shapes do not match along the specified axes pairwise in
     *                                  {@code aAxes} and {@code bAxes}.
     * @throws IllegalArgumentException If {@code aAxes} and {@code bAxes} do not match in length, or if any of the axes
     *                                  are out of bounds for the corresponding tensor.
     */
    @Override
    public AbstractNDArray<?, Complex128[], Complex128> tensorDot(CooCMatrix src2, int[] aAxes, int[] bAxes) {
        return toTensor().tensorDot(src2.toTensor(), aAxes, bAxes);
    }


    /**
     * Constructs a sparse CSR matrix of a similar type to this sparse COO matrix.
     *
     * @param shape Shape of the CSR matrix to construct.
     * @param entries Non-zero data of the CSR matrix.
     * @param rowPointers Non-zero row pointers of the CSR matrix.
     * @param colIndices Non-zero column indices of the CSR matrix.
     *
     * @return A CSR matrix of a similar type to this sparse COO matrix.
     */
    @Override
    public CsrCMatrix makeLikeCsrMatrix(Shape shape, Complex128[] entries, int[] rowPointers, int[] colIndices) {
        return new CsrCMatrix(shape, entries, rowPointers, colIndices);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        return CooRingMatrixOps.isHermitian(shape, data, rowIndices, colIndices);
    }


    /**
     * Converts this sparse COO matrix to an equivalent sparse CSR matrix.
     *
     * @return A sparse CSR matrix equivalent to this sparse COO matrix.
     */
    @Override
    public CsrCMatrix toCsr() {
        return (CsrCMatrix) super.toCsr();
    }


    /**
     * Converts this matrix to an equivalent tensor.
     *
     * @return A tensor which is equivalent to this matrix.
     */
    @Override
    public CooCTensor toTensor() {
        int[][] indices = {rowIndices.clone(), colIndices.clone()};
        indices = RealDenseTranspose.blockedIntMatrix(indices);
        return CooCTensor.unsafeMake(shape, data.clone(), indices);
    }


    /**
     * Converts this matrix to an equivalent tensor with the specified shape.
     *
     * @param newShape New shape for the tensor. Can be any rank but must have the same number of total entries as {@link #shape this.shape}.
     *
     * @return A tensor equivalent to this matrix which has been reshaped to {@code newShape}
     */
    @Override
    public CooCTensor toTensor(Shape newShape) {
        return toTensor().reshape(newShape);
    }


    /**
     * Computes the matrix-vector multiplication of a vector with this matrix.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of multiplying this matrix with {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix do not equal the size of
     *                                {@code b}.
     */
    @Override
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(shape, b.shape);
        Complex128[] dest = new Complex128[b.size];
        CooSemiringMatMult.standardVector(
                data, rowIndices, colIndices, shape,
                b.data, b.indices, dest);

        return new CVector(dest);
    }


    /**
     * Augments a vector to this matrix. That is, inserts the vector as a new column on the right hand side of the matrix.
     * @param b The vector to augment to this matrix.
     * @return A new matrix resulting from augmenting the vector {@code b} to this matrix.
     * @throws IllegalArgumentException If {@code a.numRows != b.size}.
     */
    public CooCMatrix augment(CooVector b) {
        return RealComplexCooConcats.augment(this, b);
    }


    /**
     * Augments a real COO matrix to this matrix. That is, combines the columns of this matrix and {@code a} into a single matrix.
     * @param a Matrix to augment to this matrix. Must have the same number of columns as this matrix.
     * @return A matrix resulting from augmenting {@code a} to this matrix. Will have shape
     * {@code (this.numRows, this.numCols + a.numCols)}.
     * @throws IllegalArgumentException If {@code this.numRows != a.numRows}
     */
    public CooCMatrix augment(CooMatrix a) {
        return RealComplexCooConcats.augment(this, a);
    }


    /**
     * Converts this matrix to an equivalent real matrix. This is done by ignoring the imaginary components of this matrix.
     * @return A real matrix which is equivalent to this matrix.
     */
    public CooMatrix toReal() {
        return CooMatrix.unsafeMake(shape, Complex128Ops.toReal(data),
                rowIndices.clone(), colIndices.clone());
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
     * Rounds all data within this matrix to the specified precision.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A new matrix containing the data of this matrix rounded to the specified precision.
     */
    public CooCMatrix round(int precision) {
        return unsafeMake(shape, Complex128Ops.round(data, precision), rowIndices.clone(), colIndices.clone());
    }


    /**
     * Sets all elements of this matrix to zero if they are within {@code tol} of zero. This is <em>not</em> done in-place.
     * @param precision The precision to round to (i.e., the number of decimal places to round to). Must be non-negative.
     * @return A copy of this matrix with all data within {@code tol} of zero set to zero.
     */
    public CooCMatrix roundToZero(double tolerance) {
        Complex128[] rounded = Complex128Ops.roundToZero(data, tolerance);
        List<Complex128> dest = new ArrayList<>(data.length);
        List<Integer> destRowIndices = new ArrayList<>(data.length);
        List<Integer> destColIndices = new ArrayList<>(data.length);

        for(int i = 0, size = data.length; i<size; i++) {
            if(!rounded[i].isZero()) {
                dest.add(rounded[i]);
                destRowIndices.add(rowIndices[i]);
                destColIndices.add(colIndices[i]);
            }
        }

        return new CooCMatrix(shape, dest, destRowIndices, destColIndices);
    }


    /**
     * Sets an index of this matrix to the specified value.
     *
     * @param value Value to set.
     * @param row Row index to set.
     * @param col Column index to set.
     *
     * @return A reference to this matrix.
     */
    public CooCMatrix set(double value, int row, int col) {
        return super.set(new Complex128(value), row, col);
    }


    /**
     * Computes the element-wise difference to two matrices.
     * @param b The second matrix in the element-wise difference.
     * @return The result of computing the element-wise difference between this matrix and {@code b}.
     */
    public CooCMatrix sub(CooMatrix b) {
        return RealComplexSparseMatOps.sub(this, b);
    }


    /**
     * Computes the element-wise multiplication between this tensor and a real COO matrix.
     * @param b Second matrix in the element-wise product.
     * @return The element-wise product of this tensor with {@code b}.
     * @throws ArrayShapeException If {@code !this.shape.equals(b.shape)}.
     */
    public CooCMatrix elemMult(CooMatrix b) {
        return RealComplexSparseMatOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise multiplication between two matrices of the same shape.
     * @param b The second matrix in the element-wise product.
     * @return The element-wise product of this tensor with {@code b}.
     */
    public CooCMatrix elemMult(Matrix b) {
        return RealComplexDenseCooMatOps.elemMult(b, this);
    }


    /**
     * Computes the element-wise multiplication between two matrices of the same shape.
     * @param b Second tensor in the element-wise product.
     * @return The element-wise product between this matrix and {@code b}.
     * @throws ArrayShapeException If {@code !this.shape.equals(b.shape)}
     */
    public CooCMatrix elemMult(CMatrix b) {
        Complex128[] dest = new Complex128[nnz];
        DenseCooFieldMatrixOps.elemMult(b.shape, b.data, shape, data, rowIndices, colIndices, dest);
        return unsafeMake(shape, dest, rowIndices.clone(), colIndices.clone());
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
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link CooCMatrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;
        return CooSemiringEquals.cooMatrixEquals(this, ((CooCMatrix) object));
    }


    @Override
    public int hashCode() {
        // Ignores explicit zeros to maintain contract with equals method.
        int result = 17;
        result = 31*result + shape.hashCode();

        for (int i = 0; i < data.length; i++) {
            if (!data[i].isZero()) {
                result = 31*result + data[i].hashCode();
                result = 31*result + Integer.hashCode(rowIndices[i]);
                result = 31*result + Integer.hashCode(colIndices[i]);
            }
        }

        return result;
    }


    /**
     * Formats this sparse matrix as a human-readable string.
     * @return A human-readable string representing this sparse matrix.
     */
    public String toString() {
        int size = nnz;
        StringBuilder result = new StringBuilder(String.format("shape: %s\n", shape));
        result.append("nnz: ").append(nnz).append("\n");
        result.append("Non-zero data: [");

        int maxCols = PrintOptions.getMaxColumns();
        boolean centering = PrintOptions.useCentering();
        int padding = PrintOptions.getPadding();
        int precision = PrintOptions.getPrecision();

        int stopIndex = Math.min(maxCols -1, size-1);
        int width;
        String value;

        if(data.length > 0) {
            // Get data up until the stopping point.
            for(int i = 0; i<stopIndex; i++) {
                value = StringUtils.ValueOfRound(data[i], precision);
                width = padding + value.length();
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            if(stopIndex < size-1) {
                width = padding + 3;
                value = "...";
                value = centering ? StringUtils.center(value, width) : value;
                result.append(String.format("%-" + width + "s", value));
            }

            // Get last entry now
            value = StringUtils.ValueOfRound(data[size-1], precision);
            width = padding + value.length();
            value = centering ? StringUtils.center(value, width) : value;
            result.append(String.format("%-" + width + "s", value));
        }

        result.append("]\n");
        result.append("Row Indices: ")
                .append(PrettyPrint.abbreviatedArray(rowIndices, maxCols, padding, centering))
                .append("\n");
        result.append("Col Indices: ")
                .append(PrettyPrint.abbreviatedArray(colIndices, maxCols, padding, centering));

        return result.toString();
    }
}
