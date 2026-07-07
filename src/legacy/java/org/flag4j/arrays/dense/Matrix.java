/*
 * MIT License
 *
 * Copyright (c) 2024-2026. Jacob Watters
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
import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.primitive_arrays.AbstractDenseDoubleNDArray;
import org.flag4j.arrays.backend.smart_visitors.MatrixVisitor;
import org.flag4j.arrays.sparse.*;
import org.flag4j.io.PrettyPrint;
import org.flag4j.linalg.MatrixNorms;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.ops.MatrixMultiplyDispatcher;
import org.flag4j.linalg.ops.RealDenseMatrixMultiplyDispatcher;
import org.flag4j.linalg.ops.TransposeDispatcher;
import org.flag4j.linalg.ops.common.complex.Complex128Ops;
import org.flag4j.linalg.ops.common.field_ops.FieldOps;
import org.flag4j.linalg.ops.dense.real.RealDenseDeterminant;
import org.flag4j.linalg.ops.dense.real.RealDenseEquals;
import org.flag4j.linalg.ops.dense.real.RealDenseProperties;
import org.flag4j.linalg.ops.dense.real.RealDenseSetOps;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseElemDiv;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseElemMult;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.real.RealDenseSparseMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_complex.RealComplexDenseCooMatOps;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatMult;
import org.flag4j.linalg.ops.dense_sparse.coo.real_field_ops.RealFieldDenseCooMatrixOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real.RealCsrDenseMatMult;
import org.flag4j.linalg.ops.dense_sparse.csr.real.RealCsrDenseOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_complex.RealComplexCsrDenseOps;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.linalg.ops.dispatch.ReDeMatMultDispatcher;
import org.flag4j.linalg.ops.dispatch.ReDeMatVecMultDispatcher;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayConversions;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * <p>Instances of this class represent a complex dense matrix backed by a {@code double[]} array.
 * The {@code Matrix} class
 * provides functionality for real dense matrix operations, supporting mutable items with a fixed shape.
 * This class extends
 * {@link AbstractDenseDoubleNDArray} and offers additional methods optimized for complex
 * arithmetic and matrix computations.
 *
 * <p>A {@code Matrix} is essentially equivalent to a rank-2 tensor but includes extended functionality
 * and may offer improved performance for certain operations compared to general rank-n tensors.
 *
 * <p><b>Key Features:</b>
 * <ul>
 *   <li>Support for standard matrix operations like addition, subtraction, multiplication, and exponentiation.</li>
 *   <li>Conversion methods to other matrix representations, such as {@link CooMatrix COO} (Coordinate) and {@link CsrMatrix CSR}
 *   (Compressed Sparse Row) formats.</li>
 *   <li>Utility methods for checking properties like being orthogonal, symmetric, etc.</li>
 * </ul>
 *
 * <p><b>Example Usage:</b>
 * <pre>{@code
 * // Constructing a complex matrix from a 2D array of complex numbers
 * double[][] items = {
 *     { 1, 2, 3 },
 *     { 4, 5, 6 },
 *     { 7, 8, 9,}};
 * Matrix matrix = new Matrix(items);
 *
 * // Performing matrix multiplication.
 * Matrix result = matrix.mult(matrix);
 *
 * // Performing matrix transpose.
 * Matrix transpose = matrix.T();
 *
 * // Performing matrix transpose.
 * Matrix conjugateTranspose = matrix.T();
 *
 * // Checking if the matrix is orthogonal.
 * boolean isUnitary = matrix.isOrthogonal();
 * }</pre>
 *
 * @see Tensor
 * @see Vector
 */
public class Matrix extends AbstractDenseDoubleNDArray<Matrix>
        implements MatrixMixin<Matrix, Matrix, Vector, Double> {
    private static final long serialVersionUID = 1L;

    // TODO: Add norm methods.
    // TODO: Add inv() method to this class and CMatrix.

    /**
     * The number of rows in this matrix.
     */
    public final int numRows;
    /**
     * The number of columns in this matrix.
     */
    public final int numCols;

    /**
     * Creates a tensor with the specified items and shape.
     *
     * @param shape Shape of this tensor.
     * @param entries Entries of this tensor. If this tensor is dense, this specifies all items within the tensor.
     * If this tensor is sparse, this specifies only the non-zero items of the tensor.
     */
    public Matrix(Shape shape, double... entries) {
        super(shape, entries);
        ValidateParameters.ensureRank(shape, 2);

        numRows = shape.getSize(0);
        numCols = shape.getSize(1);
    }


    /**
     * Constructs a square real dense matrix of a specified size. The items of the matrix will default to zero.
     * @param size Size of the square matrix.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size) {
        super(new Shape(size, size), new double[size*size]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a square real dense matrix with a specified fill value.
     * @param size Size of the square matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if size negative.
     */
    public Matrix(int size, double value) {
        super(new Shape(size, size), new double[size*size]);
        Arrays.fill(super.data, value);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a real dense matrix of a specified shape filled with zeros.
     * @param rows The number of rows in the matrix.
     * @param cols The number of columns in the matrix.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols) {
        super(new Shape(rows, cols), new double[rows*cols]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a real dense matrix with a specified shape and fills the matrix with the specified value.
     * @param rows Number of rows in the matrix.
     * @param cols Number of columns in the matrix.
     * @param value Value to fill this matrix with.
     * @throws IllegalArgumentException if either m or n is negative.
     */
    public Matrix(int rows, int cols, double value) {
        super(new Shape(rows, cols), new double[rows*cols]);
        Arrays.fill(super.data, value);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a real dense matrix whose items are specified by a double array.
     * @param data Entries of the real dense matrix.
     */
    public Matrix(Double[][] data) {
        super(new Shape(data.length, data[0].length),
                new double[data.length*data[0].length]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);

        int index = 0;
        for(Double[] row : data) {
            for(Double value : row)
                super.data[index++] = value;
        }
    }


    /**
     * Creates a real dense matrix whose items are specified by a double array.
     * @param data Entries of the real dense matrix.
     */
    public Matrix(Integer[][] data) {
        super(new Shape(data.length, data[0].length),
                new double[data.length*data[0].length]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);

        int index = 0;
        for(Integer[] row : data) {
            for(Integer value : row)
                super.data[index++] = value;
        }
    }


    /**
     * Creates a real dense matrix whose items are specified by a double array.
     * @param data Entries of the real dense matrix.
     */
    public Matrix(double[][] data) {
        super(new Shape(data.length, data[0].length),
                new double[data.length*data[0].length]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);

        int index = 0;
        for(double[] row : data) {
            for(double value : row)
                super.data[index++] = value;
        }
    }


    /**
     * Creates a real dense matrix whose items are specified by a double array.
     * @param data Entries of the real dense matrix.
     */
    public Matrix(int[][] data) {
        super(new Shape(data.length, data[0].length), new double[data.length*data[0].length]);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);

        // Copy the int array
        int index=0;
        for(int[] row : data) {
            for(int value : row)
                super.data[index++] = value;
        }
    }


    /**
     * Creates a real dense matrix, which is a copy of a specified matrix.
     * @param A The matrix defining the items for this matrix.
     */
    public Matrix(Matrix A) {
        super(A.shape, A.data.clone());
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a real dense matrix with the specified shape filled with zeros.
     * @param shape Shape of matrix.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape) {
        super(shape, new double[shape.numelIntValueExact()]);
        ValidateParameters.ensureRank(shape, 2);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Creates a real dense matrix with the specified shape filled with a specific value.
     * @param shape Shape of matrix.
     * @param value Value to fill matrix with.
     * @throws IllegalArgumentException If the {@code shape} is not of rank 2.
     */
    public Matrix(Shape shape, double value) {
        super(shape, new double[shape.numel().intValue()]);
        Arrays.fill(super.data, value);
        ValidateParameters.ensureRank(shape, 2);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Constructs a matrix with specified shape and items.
     * @param numRows Number of rows in this matrix.
     * @param numCols Number of columns in this matrix.
     * @param data Entries of the matrix.
     */
    public Matrix(int numRows, int numCols, double... data) {
        super(new Shape(numRows, numCols), data);
        this.numRows = shape.getSize(0);
        this.numCols = shape.getSize(1);
    }


    /**
     * Constructs a diagonal matrix from an array specifying the diagonal elements of the matrix.
     * @param data Diagonal elements of the matrix. All other values will be zero.
     * @return A diagonal matrix whose diagonal elements are equal to {@code items}.
     * @see #diag(Vector)
     */
    public static Matrix diag(double... data) {
        int size = data.length;
        double[] fullData = new double[size*size];

        int destIdx = 0;
        for(int i=0; i<size; i++) {
            fullData[destIdx] = data[i];
            destIdx += size + 1;
        }

        return new Matrix(size, size, fullData);
    }


    /**
     * Constructs a diagonal matrix from a vector specifying the diagonal elements of the matrix.
     * @param vec Diagonal elements of the matrix. All other values will be zero.
     * @return A diagonal matrix whose diagonal elements are equal to the entries of {@code vec}.
     * @see #diag(double...)
     */
    public static Matrix diag(Vector vec) {
        return diag(vec.data);
    }


    /**
     * Constructs a tensor of the same type as this tensor with the given shape and items.
     *
     * @param shape Shape of the tensor to construct.
     * @param data Entries of the tensor to construct.
     *
     * @return A tensor of the same type as this tensor with the given shape and items.
     */
    @Override
    public Matrix makeLikeNDArray(Shape shape, double[] data) {
        return new Matrix(shape, data);
    }


    /**
     * Gets the length of the items array which backs this matrix.
     *
     * @return The length of the items array which backs this matrix.
     */
    @Override
    public int dataLength() {
        return data.length;
    }


    /**
     * Flattens this matrix to a row vector.
     *
     * @return The flattened matrix.
     *
     * @see #flatten(int)
     */
    @Override
    public Matrix flatten() {
        return flatten(1);
    }
    

    /**
     * Flattens this matrix along the specified axis.
     *
     * @param axis Axis along which to flatten tensor.
     * @return A new matrix containing the same items as this matrix but flattened along the specified axis.
     * <ul>
     *     <li>If {@code axis == 0} a matrix with the shape {@code (this.numRows*this.numCols, 1)} is returned.</li>
     *     <li>If {@code axis == 1} a matrix with the shape {@code (1, this.numRows*this.numCols)} is returned.</li>
     * </ul>
     * @throws ArrayIndexOutOfBoundsException If the axis is negative or larger than {@code this.{@link #getRank()}-1}.
     * @see #flatten()
     */
    @Override
    public Matrix flatten(int axis) {
        ValidateParameters.ensureValidAxes(shape, axis);
        return (axis == 0)
                ? new Matrix(data.length, 1, data.clone())
                : new Matrix(1, data.length, data.clone());
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
    public static Matrix I(int size) {
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
    public static Matrix I(int numRows, int numCols) {
        ValidateParameters.ensureNonNegative(numRows, numCols);
        Matrix I = new Matrix(numRows, numCols);
        int stop = Math.min(numRows, numCols);

        for(int i=0; i<stop; i++)
            I.data[i*numCols+i] = 1.0;

        return I;
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
    public static Matrix I(Shape shape) {
        ValidateParameters.ensureRank(shape, 2);
        return I(shape.getSize(0), shape.getSize(1));
    }


    /**
     * Gets the number of rows in this matrix.
     *
     * @return The number of rows in this matrix.
     */
    @Override
    public int numRows() {
        return numRows;
    }


    /**
     * Gets the number of columns in this matrix.
     *
     * @return The number of columns in this matrix.
     */
    @Override
    public int numCols() {
        return numCols;
    }


    /**
     * Gets the element of this matrix at this specified {@code row} and {@code col}.
     *
     * @param row Row index of the item to get from this matrix.
     * @param col Column index of the item to get from this matrix.
     *
     * @return The element of this matrix at the specified index.
     */
    @Override
    public Double get(int row, int col) {
        return data[row*numCols + col];
    }


    /**
     * <p>Computes the trace of this matrix. That is, the sum of elements along the principle diagonal of this matrix.
     *
     * <p>Same as {@link #trace()}.
     *
     * @return The trace of this matrix.
     *
     * @throws IllegalArgumentException If this matrix is not square.
     */
    @Override
    public Double tr() {
        ValidateParameters.ensureSquareMatrix(this.shape);
        double sum = 0;
        int colsOffset = this.numCols+1;

        for(int i=0; i<this.numRows; i++)
            sum += this.data[i*colsOffset];

        return sum;
    }


    /**
     * Checks if this matrix is upper triangular.
     *
     * @return {@code true} is this matrix is upper triangular; {@code false} otherwise.
     *
     * @see #isTri()
     * @see #isTriL()
     * @see #isDiag()
     */
    @Override
    public boolean isTriU() {
        if(!isSquare()) return false;

        // Ensure the lower half is zeros.
        for(int i=1; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j<i; j++)
                if(data[rowOffset + j] != 0) return false; // No need to continue.
        }

        return true;
    }


    /**
     * Checks if this matrix is lower triangular.
     *
     * @return {@code true} is this matrix is lower triangular; {@code false} otherwise.
     *
     * @see #isTri()
     * @see #isTriU()
     * @see #isDiag()
     */
    @Override
    public boolean isTriL() {
        if(!isSquare()) return false;

        // Ensure upper half is zeros.
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=i+1; j<numCols; j++)
                if(data[rowOffset + j] != 0) return false; // No need to continue.
        }

        return true; // If we reach this point, the matrix is lower triangular.
    }


    /**
     * Checks if this matrix is the identity matrix. That is, checks if this matrix is square and contains
     * only ones along the principle diagonal and zeros everywhere else.
     *
     * @return {@code true} if this matrix is the identity matrix; {@code false} otherwise.
     */
    @Override
    public boolean isI() {
        return RealDenseProperties.isIdentity(this);
    }


    /**
     * Checks that this matrix is close to the identity matrix.
     *
     * @return True if this matrix is approximately the identity matrix.
     *
     * @see #isI()
     */
    public boolean isCloseToI() {
        return RealDenseProperties.isCloseToIdentity(this);
    }


    /**
     * Computes the determinant of a square matrix.
     *
     * @return The determinant of this matrix.
     *
     * @throws LinearAlgebraException If this matrix is not square.
     */
    public Double det() {
        return RealDenseDeterminant.det(this);
    }


    /**
     * <p>Computes the rank of this matrix (i.e., the number of linearly independent rows/columns in this matrix).
     *
     * <p>This is computed as the number of singular values greater than {@code tol} where:
     * <pre>{@code double tol = 2.0*Math.max(rows, cols)*Flag4jConstants.EPS_F64*Math.min(this.numRows, this.numCols);}</pre>
     *
     *
     * <p>Note the "matrix rank" is <b>NOT</b> related to the "{@link AbstractNDArray#getRank() tensor rank}" which
     * is number of indices
     * needed to uniquely specify an entry in the tensor.
     *
     * @return The matrix rank of this matrix.
     */
    public int matrixRank() {
        return new RealSVD(false).decompose(this).getRank();
    }


    /**
     * Computes the matrix multiplication between two matrices.
     *
     * @param b Second matrix in the matrix multiplication.
     *
     * @return The result of matrix multiplying this matrix with matrix {@code b}.
     *
     * @throws LinearAlgebraException If the number of columns in this matrix does not equal
     * the number of rows in matrix {@code b}.
     */
    @Override
    public Matrix mult(Matrix b) {
        return ReDeMatMultDispatcher.dispatch(this, b);
    }


    /**
     * Multiplies this matrix with the transpose of the {@code b} tensor as if by
     * {@code this.mult(b.T())}.
     * For large matrices, this method may
     * be significantly faster than directly computing the transpose followed by the multiplication as
     * {@code this.mult(b.T())}.
     *
     * @param b The second matrix in the multiplication and the matrix to transpose/
     *
     * @return The result of multiplying this matrix with the transpose of {@code b}.
     */
    @Override
    public Matrix multTranspose(Matrix b) {
        // Ensure this matrix can be multiplied to the transpose of B.
        ValidateParameters.ensureAllEqual(this.numCols, b.numCols);

        return new Matrix(
                new Shape(this.numRows, b.numRows),
                RealDenseMatrixMultiplyDispatcher.dispatchTranspose(this, b)
        );
    }


    /**
     * Computes the Frobenius inner product of two matrices.
     *
     * @param b Second matrix in the Frobenius inner product
     *
     * @return The Frobenius inner product of this matrix and matrix B.
     *
     * @throws IllegalArgumentException If this matrix and B have different shapes.
     */
    @Override
    public Double fib(Matrix b) {
        ValidateParameters.ensureEqualShape(this.shape, b.shape);
        return this.T().mult(b).tr();
    }


    /**
     * Stacks matrices along columns.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking this matrix on top of the matrix {@code b}.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of columns.
     * @see #stack(MatrixMixin, int) 
     * @see #augment(Matrix)
     */
    @Override
    public Matrix stack(Matrix b) {
        ValidateParameters.ensureArrayLengthsEq(this.numCols, b.numCols);
        Matrix stacked = new Matrix(new Shape(this.numRows + b.numRows, this.numCols));

        System.arraycopy(this.data, 0, stacked.data, 0, this.data.length);
        System.arraycopy(b.data, 0, stacked.data, this.data.length, b.data.length);

        return stacked;
    }


    /**
     * Stacks matrices along rows.
     *
     * @param b Matrix to stack to this matrix.
     *
     * @return The result of stacking {@code b} to the right of this matrix.
     *
     * @throws IllegalArgumentException If this matrix and matrix {@code b} have a different number of rows.
     * @see #stack(Matrix)
     * @see #stack(MatrixMixin, int) 
     */
    @Override
    public Matrix augment(Matrix b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.numRows);
        Matrix augmented = new Matrix(new Shape(numRows, numCols + b.numCols));

        // Copy items from two matrices.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(data, i*numCols, augmented.data, i*augmented.numCols, numCols);

            int augmentedRowOffset = i*augmented.numCols + numCols;
            int bRowOffset = i*b.numCols;
            for(int j=0, cols=b.numCols; j<cols; j++)
                augmented.data[augmentedRowOffset + j] = b.data[bRowOffset + j];
        }

        return augmented;
    }


    /**
     * Augments a vector to this matrix.
     *
     * @param b The vector to augment to this matrix.
     *
     * @return The result of augmenting {@code b} to this matrix.
     */
    @Override
    public Matrix augment(Vector b) {
        ValidateParameters.ensureArrayLengthsEq(numRows, b.size);
        Matrix augmented = new Matrix(new Shape(numRows, numCols + 1));

        // Copy items from this matrix.
        for(int i=0; i<numRows; i++) {
            System.arraycopy(data, i*numCols, augmented.data, i*augmented.numCols, numCols);
            augmented.data[i*augmented.numCols + numCols] = b.data[i];
        }

        return augmented;
    }


    /**
     * Swaps specified rows in the matrix. This is done in-place.
     *
     * @param rowIndex1 Index of the first row to swap.
     * @param rowIndex2 Index of the second row to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public Matrix swapRows(int rowIndex1, int rowIndex2) {
        ValidateParameters.ensureAllGreaterEq(0, rowIndex1, rowIndex2);
        ValidateParameters.ensureGreaterEq(rowIndex1, this.numRows-1);
        ValidateParameters.ensureGreaterEq(rowIndex2, this.numRows-1);

        double temp;
        int row1Start = rowIndex1*numCols;
        int row2Start = rowIndex2*numCols;
        int stop = row1Start + numCols;

        while(row1Start < stop) {
            ArrayUtils.swap(data, row1Start++, row2Start++);
        }

        return this;
    }


    /**
     * Swaps specified columns in the matrix. This is done in-place.
     *
     * @param colIndex1 Index of the first column to swap.
     * @param colIndex2 Index of the second column to swap.
     *
     * @return A reference to this matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If either index is outside the matrix bounds.
     */
    @Override
    public Matrix swapCols(int colIndex1, int colIndex2) {
        ValidateParameters.ensureAllGreaterEq(0, colIndex1, colIndex2);
        ValidateParameters.ensureGreaterEq(colIndex1, this.numCols-1);
        ValidateParameters.ensureGreaterEq(colIndex2, this.numCols-1);

        double temp;
        for(int i=0; i<numRows; i++) {
            // Swap elements.
            int idx = i*numCols;
            ArrayUtils.swap(data, idx + colIndex1, idx + colIndex2);
        }

        return this;
    }


    /**
     * Checks if a matrix is symmetric. That is, if the matrix is square and equal to its transpose.
     *
     * @return {@code true} if this matrix is symmetric; {@code false} otherwise.
     */
    @Override
    public boolean isSymmetric() {
        return RealDenseProperties.isSymmetric(data, shape);
    }


    /**
     * Checks if a matrix is Hermitian. That is, if the matrix is square and equal to its conjugate transpose.
     *
     * @return {@code true} if this matrix is Hermitian; {@code false} otherwise.
     */
    @Override
    public boolean isHermitian() {
        return isSymmetric();
    }


    /**
     * Checks if this matrix is orthogonal. That is, if the inverse of this matrix is equal to its transpose.
     *
     * @return {@code true} if this matrix it is orthogonal; {@code false} otherwise.
     */
    @Override
    public boolean isOrthogonal() {
        return numRows == numCols && RealDenseProperties.isCloseToIdentity(multTranspose(this));
    }


    /**
     * Removes a specified row from this matrix.
     *
     * @param rowIndex Index of the row to remove from this matrix.
     *
     * @return A copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRow(int rowIndex) {
        Matrix copy = new Matrix(this.numRows-1, this.numCols);

        int row = 0;
        for(int i=0; i<this.numRows; i++) {
            if(i != rowIndex) {
                System.arraycopy(this.data, i*numCols, copy.data, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of rows from this matrix.
     *
     * @param rowIndices The indices of the rows to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeRows(int... rowIndices) {
        Matrix copy = new Matrix(this.numRows-rowIndices.length, this.numCols);

        int row = 0;

        for(int i=0; i<this.numRows; i++) {
            if(ArrayUtils.notContains(rowIndices, i)) {
                System.arraycopy(this.data, i*numCols, copy.data, row*copy.numCols, this.numCols);
                row++;
            }
        }

        return copy;
    }


    /**
     * Removes a specified column from this matrix.
     *
     * @param colIndex Index of the column to remove from this matrix.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCol(int colIndex) {
        Matrix copy = new Matrix(this.numRows, this.numCols-1);

        int col;
        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(j!=colIndex) {
                    copy.data[i*copy.numCols + col] = this.data[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
    }


    /**
     * Removes a specified set of columns from this matrix.
     *
     * @param colIndices Indices of the columns to remove from this matrix. Assumed to contain unique values.
     *
     * @return a copy of this matrix with the specified column removed.
     */
    @Override
    public Matrix removeCols(int... colIndices) {
        Matrix copy = new Matrix(this.numRows, this.numCols - colIndices.length);

        int col;

        for(int i=0; i<this.numRows; i++) {
            col = 0;
            for(int j=0; j<this.numCols; j++) {
                if(ArrayUtils.notContains(colIndices, j)) {
                    copy.data[i*copy.numCols + col] = this.data[i*numCols + j];
                    col++;
                }
            }
        }

        return copy;
    }




    /**
     * Creates a copy of this matrix and sets a slice of the copy to the specified values. The rowStart and colStart parameters specify the upper
     * left index location of the slice to set.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A copy of this matrix with the given slice set to the specified values.
     *
     * @throws IndexOutOfBoundsException If rowStart or colStart are not within the matrix.
     * @throws IllegalArgumentException  If the {@code values} slice, with the upper-left corner at the specified location, does not
     *                                   fit completely within this matrix.
     */
    @Override
    public Matrix setSliceCopy(Matrix values, int rowStart, int colStart) {
        ValidateParameters.validateArrayIndices(numRows, rowStart);
        ValidateParameters.validateArrayIndices(numCols, colStart);
        Matrix copy = new Matrix(this);

        for(int i=0; i<values.numRows; i++) {
            System.arraycopy(
                    values.data, i*values.numCols,
                    copy.data, (i+rowStart)*numCols + colStart, values.numCols
            );
        }

        return copy;
    }


    /**
     * Sets a slice of this matrix to the specified {@code values}. The {@code rowStart} and {@code colStart} parameters specify the
     * upper left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException If the {@code values} slice, with the upper-left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    public Matrix setSlice(Matrix values, int rowStart, int colStart) {
        ValidateParameters.validateArrayIndices(numRows, rowStart);
        ValidateParameters.validateArrayIndices(numCols, colStart);

        for(int i=0; i<values.numRows; i++) {
            System.arraycopy(
                    values.data, i*values.numCols,
                    data, (i+rowStart)*numCols + colStart, values.numCols
            );
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified {@code values}. The {@code rowStart} and {@code colStart} parameters specify the
     * upper left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException If the {@code values} slice, with the upper-left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    public Matrix setSlice(double[][] values, int rowStart, int colStart) {
        ValidateParameters.ensureLessEq(numRows, rowStart + values.length);
        ValidateParameters.ensureLessEq(numCols, colStart + values[0].length);
        ValidateParameters.ensureAllGreaterEq(0, rowStart, colStart);
        int cols = values[0].length;

        for(int i=0, size=values.length; i<size; i++) {
            int rowOffset = (i+rowStart)*numCols + colStart;
            double[] row = values[i];

            for(int j=0; j<cols; j++)
                this.data[rowOffset + j] = row[j];
        }

        return this;
    }


    /**
     * Sets a slice of this matrix to the specified {@code values}. The {@code rowStart} and {@code colStart} parameters specify the
     * upper left index location of the slice to set within this matrix.
     *
     * @param values New values for the specified slice.
     * @param rowStart Starting row index for the slice (inclusive).
     * @param colStart Starting column index for the slice (inclusive).
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If {@code rowStart} or {@code colStart} are not within the matrix.
     * @throws IllegalArgumentException If the {@code values} slice, with the upper-left corner at the specified location, does not
     *                                  fit completely within this matrix.
     */
    public Matrix setSlice(Double[][] values, int rowStart, int colStart) {
        ValidateParameters.ensureLessEq(numRows, rowStart + values.length);
        ValidateParameters.ensureLessEq(numCols, colStart + values[0].length);
        ValidateParameters.ensureAllGreaterEq(0, rowStart, colStart);

        for(int i=0, size=values.length; i<size; i++) {
            int rowOffset = (i+rowStart)*numCols + colStart;
            Double[] row = values[i];

            for(int j=0; j<values[0].length; j++)
                this.data[rowOffset + j] = row[j];
        }

        return this;
    }


    /**
     * Gets a specified slice of this matrix.
     *
     * @param rowStart Starting row index of slice (inclusive).
     * @param rowEnd Ending row index of slice (exclusive).
     * @param colStart Starting column index of slice (inclusive).
     * @param colEnd Ending row index of slice (exclusive).
     *
     * @return The specified slice of this matrix. This is a completely new matrix and <b>NOT</b> a view into the matrix.
     *
     * @throws ArrayIndexOutOfBoundsException If any of the indices are out of bounds of this matrix.
     * @throws IllegalArgumentException       If {@code rowEnd} is not greater than {@code rowStart} or if {@code colEnd} is not greater than {@code colStart}.
     */
    @Override
    public Matrix getSlice(int rowStart, int rowEnd, int colStart, int colEnd) {
        Matrix slice = new Matrix(rowEnd - rowStart, colEnd - colStart);

        for(int i=0; i<slice.numRows; i++) {
            System.arraycopy(
                    this.data, (i+rowStart)*this.numCols + colStart,
                    slice.data, i*slice.numCols, slice.numCols);
        }

        return slice;
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
    @Override
    public Matrix set(Double value, int row, int col) {
        data[row*numCols + col] = value;
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public Matrix setValues(Double[][] values) {
        ValidateParameters.ensureEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOps.setValues(values, data);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public Matrix setValues(double[][] values) {
        ValidateParameters.ensureEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOps.setValues(values, data);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public Matrix setValues(Integer[][] values) {
        ValidateParameters.ensureEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOps.setValues(values, data);
        return this;
    }


    /**
     * Sets the value of this matrix using a 2D array.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different shape then this matrix.
     */
    public Matrix setValues(int[][] values) {
        ValidateParameters.ensureEqualShape(shape, new Shape(values.length, values[0].length));
        RealDenseSetOps.setValues(values, data);
        return this;
    }


    /**
     * Sets the value of this matrix using another matrix.
     *
     * @param values New values of the matrix.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} matrix has a different shape then this matrix.
     */
    public Matrix setValues(Matrix values) {
        ValidateParameters.ensureEqualShape(shape, values.shape);
        RealDenseSetOps.setValues(values.data, data);
        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of rows in this matrix.
     */
    @Override
    public Matrix setCol(Vector values, int colIndex) {
        return setCol(values.data, colIndex);
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different length than the number of rows in this matrix.
     */
    public Matrix setCol(Double[] values, int colIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.length, this.numRows);

        int rowOffset = 0;
        for(int i=0; i<values.length; i++) {
            super.data[rowOffset + colIndex] = values[i];
            rowOffset += numCols;
        }

        return this;
    }


    /**
     * Sets a column of this matrix at the given index to the specified values.
     *
     * @param values New values for the column.
     * @param colIndex The index of the column which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different length than the number of rows in this matrix.
     */
    public Matrix setCol(double[] values, int colIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.length, this.numRows);

        int rowOffset = 0;
        for(int i=0; i<values.length; i++) {
            super.data[rowOffset + colIndex] = values[i];
            rowOffset += numCols;
        }

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} vector has a different length than the number of rows in this matrix.
     */
    @Override
    public Matrix setRow(Vector values, int rowIndex) {
        return setRow(values.data, rowIndex);
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different length than the number of rows in this matrix.
     */
    public Matrix setRow(double[] values, int rowIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.length, this.numCols);

        for(int i=0, size=values.length, rowOffset=rowIndex*numCols; i<size; i++)
            super.data[rowOffset + i] = values[i];

        return this;
    }


    /**
     * Sets a row of this matrix at the given index to the specified values.
     *
     * @param values New values for the row.
     * @param rowIndex The index of the row which is to be set.
     *
     * @return A reference to this matrix.
     *
     * @throws IllegalArgumentException If the {@code values} array has a different length than the number of rows in this matrix.
     */
    public Matrix setRow(Double[] values, int rowIndex) {
        ValidateParameters.ensureArrayLengthsEq(values.length, this.numCols);

        for(int i=0, size=values.length, rowOffset=rowIndex*numCols; i<size; i++)
            super.data[rowOffset + i] = values[i];

        return this;
    }


    /**
     * Computes the element-wise square root of a tensor.
     *
     * @return The result of applying an element-wise square root to this tensor. Note, this method will compute
     * the principle square root, i.e., the square root with a positive real part.
     */
    public CMatrix sqrtComplex() {
        return new CMatrix(shape, Complex128Ops.sqrt(data));
    }


    /**
     * Extracts the upper-triangular portion of this matrix with a specified diagonal offset. All other items of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for an upper-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all items at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all items at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all items at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The upper-triangular portion of this matrix with a specified diagonal offset. All other items of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public Matrix getTriU(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Matrix result = new Matrix(numRows, numCols);

        // Extract the upper triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=Math.max(0, i + diagOffset); j<numCols; j++) {
                if (j >= i + diagOffset) {
                    result.data[rowOffset + j] = this.data[rowOffset + j];
                }
            }
        }

        return result;
    }


    /**
     * Extracts the lower-triangular portion of this matrix with a specified diagonal offset. All other items of the resulting
     * matrix will be zero.
     *
     * @param diagOffset Diagonal offset for a lower-triangular portion to extract:
     * <ul>
     *     <li>If zero, then all items at and above the principle diagonal of this matrix are extracted.</li>
     *     <li>If positive, then all items at and above the equivalent super-diagonal are extracted.</li>
     *     <li>If negative, then all items at and above the equivalent sub-diagonal are extracted.</li>
     * </ul>
     *
     * @return The lower-triangular portion of this matrix with a specified diagonal offset. All other items of the returned
     * matrix will be zero.
     *
     * @throws IllegalArgumentException If {@code diagOffset} is not in the range (-numRows, numCols).
     */
    @Override
    public Matrix getTriL(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -numRows+1, numCols-1, "diagOffset");
        Matrix result = new Matrix(numRows, numCols);

        // Extract the lower triangular portion
        for(int i=0; i<numRows; i++) {
            int rowOffset = i*numCols;

            for(int j=0; j <= Math.min(numCols - 1, i + diagOffset); j++) {
                if(j <= i + diagOffset) {
                    result.data[rowOffset + j] = this.data[rowOffset + j];
                }
            }
        }

        return result;
    }


    /**
     * Computes matrix-vector multiplication.
     *
     * @param b Vector in the matrix-vector multiplication.
     *
     * @return The result of matrix multiplying this matrix with vector {@code b}.
     *
     * @throws IllegalArgumentException If the number of columns in this matrix does not equal the
     *                                  number of items in the vector {@code b}.
     */
    @Override
    public Vector mult(Vector b) {
        return ReDeMatVecMultDispatcher.dispatch(this, b);
    }


    /**
     * <p>Converts this matrix to an equivalent vector.
     *
     * <p>If this matrix is not shaped as a row/column vector, it will first be flattened, then converted to a vector.
     *
     * @return A vector equivalent to this matrix.
     */
    @Override
    public Vector toVector() {
        return new Vector(data.clone());
    }


    /**
     * Converts this matrix to an equivalent {@link Tensor}.
     * @return A {@link Tensor} which is equivalent to this matrix.
     */
    public Tensor toTensor() {
        return new Tensor(shape, data.clone());
    }


    /**
     * Gets a specified row of this matrix between {@code colStart} (inclusive) and {@code colEnd} (exclusive).
     *
     * @param rowIdx Index of the row in this matrix to get.
     * @param colStart Starting column of the row (inclusive).
     * @param colEnd Ending column of the row (exclusive).
     *
     * @return The row at index {@code rowIdx} of this matrix between the {@code colStart} and {@code colEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException  If {@code colEnd} is less than {@code colStart}.
     */
    @Override
    public Vector getRow(int rowIdx, int colStart, int colEnd) {
        ValidateParameters.validateArrayIndices(numCols, colStart, colEnd-1);
        ValidateParameters.ensureGreaterEq(colStart, colEnd);
        int start = rowIdx*numCols + colStart;
        int stop = rowIdx*numCols + colEnd;

        double[] row = Arrays.copyOfRange(this.data, start, stop);

        return new Vector(row);
    }


    /**
     * Gets a specified column of this matrix between {@code rowStart} (inclusive) and {@code rowEnd} (exclusive).
     *
     * @param colIdx Index of the column in this matrix to get.
     * @param rowStart Starting row of the column (inclusive).
     * @param rowEnd Ending row of the column (exclusive).
     *
     * @return The column at index {@code colIdx} of this matrix between the {@code rowStart} and {@code rowEnd}
     * indices.
     *
     * @throws IndexOutOfBoundsException If either {@code colEnd} are {@code colStart} out of bounds for the shape of this matrix.
     * @throws IllegalArgumentException If {@code rowEnd} is less than {@code rowStart}.
     */
    @Override
    public Vector getCol(int colIdx, int rowStart, int rowEnd) {
        ValidateParameters.validateArrayIndices(numRows, rowStart, rowEnd-1);
        ValidateParameters.ensureGreaterEq(rowStart, rowEnd);
        double[] col = new double[numRows];

        for(int i=rowStart; i<rowEnd; i++)
            col[i] = data[i*numCols + colIdx];

        return new Vector(col);
    }


    /**
     * Extracts the diagonal elements of this matrix and returns them as a vector.
     *
     * @return A vector containing the diagonal items of this matrix.
     */
    @Override
    public Vector getDiag() {
        int newSize = Math.min(numRows, numCols);
        double[] diag = new double[newSize];

        int idx = 0;
        for(int i=0; i<newSize; i++) {
            diag[i] = this.data[idx];
            idx += numCols + 1;
        }

        return new Vector(diag);
    }


    /**
     * Gets the elements of this matrix along the specified diagonal.
     *
     * @param diagOffset The diagonal to get within this matrix.
     * <ul>
     *     <li>If {@code diagOffset == 0}: Then the elements of the principle diagonal are collected.</li>
     *     <li>If {@code diagOffset < 0}: Then the elements of the sub-diagonal {@code diagOffset} below the principle diagonal
     *     are collected.</li>
     *     <li>If {@code diagOffset > 0}: Then the elements of the super-diagonal {@code diagOffset} above the principle diagonal
     *     are collected.</li>
     * </ul>
     *
     * @return The elements of the specified diagonal as a vector.
     */
    @Override
    public Vector getDiag(int diagOffset) {
        ValidateParameters.ensureInRange(diagOffset, -(numRows-1), numCols-1, "diagOffset");

        // Check for some quick returns.
        if(numRows == 1 && diagOffset > 0) return new Vector(data[diagOffset]);
        if(numCols == 1 && diagOffset < 0) return new Vector(data[-diagOffset]);

        // Compute the length of the diagonal.
        int newSize = Math.min(numRows, numCols);
        int idx = 0;

        if(diagOffset > 0) {
            newSize = Math.min(newSize, numCols - diagOffset);
            idx = diagOffset;
        } else if(diagOffset < 0) {
            newSize = Math.min(newSize, numRows + diagOffset);
            idx = -diagOffset*numCols;
        }

        double[] diag = new double[newSize];

        for(int i=0; i<newSize; i++) {
            diag[i] = this.data[idx];
            idx += numCols + 1;
        }

        return new Vector(diag);
    }


    /**
     * Computes the transpose of a tensor by exchanging the first and last axes of this tensor.
     *
     * @return The transpose of this tensor.
     *
     * @see #T(int, int)
     * @see #T(int...)
     */
    @Override
    public Matrix T() {
        return super.T();
    }


    /**
     * Computes the transpose of a tensor by exchanging {@code axis1} and {@code axis2}.
     *
     * @param axis1 First axis to exchange.
     * @param axis2 Second axis to exchange.
     *
     * @return The transpose of this tensor along the specified axes.
     *
     * @throws IndexOutOfBoundsException If either {@code axis1} or {@code axis2} are out of bounds for the rank of this tensor.
     * @see #T()
     * @see #T(int...)
     */
    @Override
    public Matrix T(int axis1, int axis2) {
        ValidateParameters.ensureValidAxes(shape, axis1, axis2);

        if(axis1==axis2) return copy();
        else return TransposeDispatcher.dispatch(this);
    }


    /**
     * Computes the transpose of this tensor. That is, permutes the axes of this tensor so that it matches
     * the permutation specified by {@code axes}.
     *
     * @param axes Permutation of tensor axis. If the tensor has rank {@code N}, then this must be an array of length
     * {@code N} which is a permutation of {@code {0, 1, 2, ..., N-1}}.
     *
     * @return The transpose of this tensor with its axes permuted by the {@code axes} array.
     *
     * @throws IndexOutOfBoundsException If any element of {@code axes} is out of bounds for the rank of this tensor.
     * @throws IllegalArgumentException  If {@code axes} is not a permutation of {@code {0, 1, 2, ... N-1}}.
     * @see #T(int, int)
     * @see #T()
     */
    @Override
    public Matrix T(int... axes) {
        ValidateParameters.ensureArrayLengthsEq(2, axes.length);
        ValidateParameters.ensureValidAxes(shape, axes[0], axes[1]);

        if(axes[0]==axes[1]) return copy();
        else return TransposeDispatcher.dispatch(this);
    }


    /**
     * Converts this dense tensor to an equivalent sparse COO tensor.
     *
     * @return A sparse COO tensor equivalent to this dense tensor.
     * @see #toCsr()
     */
    public CooMatrix toCoo() {
        final int rows = numRows;
        final int cols = numCols;
        List<Double> sparseEntries = new ArrayList<>();
        List<Integer> rowIndices = new ArrayList<>();
        List<Integer> colIndices = new ArrayList<>();

        for(int i=0; i<rows; i++) {
            int rowOffset = i*cols;

            for(int j=0; j<cols; j++) {
                double val = data[rowOffset + j];

                if(val != 0.0) {
                    sparseEntries.add(val);
                    rowIndices.add(i);
                    colIndices.add(j);
                }
            }
        }

        return new CooMatrix(shape, sparseEntries, rowIndices, colIndices);
    }


    /**
     * Converts this dense matrix to a sparse CSR matrix.
     * @return A sparse CSR matrix equivalent to this dense matrix.
     * @see #toCoo()
     */
    public CsrMatrix toCsr() {
        return toCoo().toCsr();
    }


    /**
     * Converts this matrix to an equivalent complex matrix.
     * @return A complex matrix with real components equal to the items of this matrix and imaginary components set to zero.
     */
    public CMatrix toComplex() {
        return new CMatrix(shape, ArrayConversions.toComplex128(data, null));
    }


    /**
     * Sums this matrix with a complex dense matrix.
     * @param b Complex dense matrix in the sum.
     * @return The element-wise sum of this matrix with {@code b}.
     */
    public CMatrix add(CMatrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.add(b.shape, b.data, shape, data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Sums this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public Matrix add(CsrMatrix b) {
        return RealCsrDenseOps.applyBinOpp(this, b, Double::sum);
    }


    /**
     * Sums this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public Matrix add(CooMatrix b) {
        return RealDenseSparseMatrixOps.add(this, b);
    }


    /**
     * Sums this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CsrCMatrix b) {
        return RealComplexCsrDenseOps.applyBinOpp(this, b, (Double x, Complex128 y)->y.add(x));
    }


    /**
     * Sums this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the sum.
     * @return The element-wise sum of this matrix and {@code b}
     */
    public CMatrix add(CooCMatrix b) {
        return RealComplexDenseCooMatOps.add(this, b);
    }


    /**
     * Adds a complex-valued scalar to each entry of this matrix.
     * @param b Scalar to add to this matrix.
     * @return A matrix containing the sum of each entry in this matrix with {@code b}.
     */
    public CMatrix add(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.add(data, b, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the difference of this matrix with a complex dense matrix.
     * @param b Complex dense matrix in the difference.
     * @return The difference of this matrix with {@code b}.
     */
    public CMatrix sub(CMatrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseOps.sub(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the difference of this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public Matrix sub(CsrMatrix b) {
        return RealCsrDenseOps.applyBinOpp(this, b, Double::sum);
    }


    /**
     * Computes the difference of this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public Matrix sub(CooMatrix b) {
        return RealDenseSparseMatrixOps.sub(this, b);
    }


    /**
     * Computes the difference of this matrix with a real sparse CSR matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CsrCMatrix b) {
        return RealComplexCsrDenseOps.applyBinOpp(this, b, (Double x, Complex128 y)->new Complex128(x-y.re, y.im));
    }


    /**
     * Computes the difference of this matrix with a real sparse COO matrix.
     * @param b real sparse CSR matrix in the difference.
     * @return The element-wise difference of this matrix and {@code b}
     */
    public CMatrix sub(CooCMatrix b) {
        return RealComplexDenseCooMatOps.sub(this, b);
    }


    /**
     * Subtracts a complex-valued scalar from each entry of this matrix.
     * @param b Scalar to subtract from each entry of this matrix.
     * @return A matrix containing the difference of each entry in this matrix with {@code b}.
     */
    public CMatrix sub(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.sub(data, b, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the matrix multiplication between this matrix and a complex dense matrix.
     * @param b The complex dense matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws NDArrayShapeException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CMatrix b) {
        Complex128[] entries = MatrixMultiplyDispatcher.dispatch(this, b);
        Shape shape = new Shape(this.numRows, b.numCols);
        return new CMatrix(shape, entries);
    }


    /**
     * Computes the matrix multiplication between this matrix and a real sparse CSR matrix.
     * @param b The real sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws NDArrayShapeException If {@code this.numCols != b.numRows}.
     */
    public Matrix mult(CsrMatrix b) {
        return RealCsrDenseMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between this matrix and a complex sparse CSR matrix.
     * @param b The complex sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws NDArrayShapeException If {@code this.numCols != b.numRows}.
     */
    public CMatrix mult(CsrCMatrix b) {
        return (CMatrix) RealFieldDenseCsrMatMult.standard(this, b);
    }


    /**
     * Computes the matrix multiplication between this matrix and a real sparse COO matrix.
     * @param b The real sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws NDArrayShapeException If {@code this.numCols != b.numRows}.
     * @implNote This method computes the matrix product as {@code this.mult(b.toCsr());}.
     */
    public Matrix mult(CooMatrix b) {
        return mult(b.toCsr());
    }


    /**
     * Computes the matrix multiplication between this matrix and a complex sparse COO matrix.
     * @param b The complex sparse matrix in the matrix multiplication.
     * @return The matrix product between this matrix and {@code b}.
     * @throws NDArrayShapeException If {@code this.numCols != b.numRows}.
     * @implNote This method computes the matrix product as {@code this.mult(b.toCsr());}.
     */
    public CMatrix mult(CooCMatrix b) {
        return mult(b.toCsr());
    }


    /**
     * Computes the matrix-vector product of this matrix and a dense complex vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public CVector mult(CVector b) {
        return new CVector(MatrixMultiplyDispatcher.dispatch(this, b));
    }


    /**
     * Computes the matrix-vector product of this matrix and a real sparse vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public Vector mult(CooVector b) {
        ValidateParameters.ensureMatMultShapes(this.shape, new Shape(b.size, 1));
        double[] entries = RealDenseSparseMatMult.standardVector(
                this.data, this.shape, b.data, b.indices);

        return new Vector(entries);
    }


    /**
     * Computes the matrix-vector product of this matrix and a complex sparse vector.
     * @param b Vector in the matrix-vector product.
     * @return The matrix-vector product of this matrix and the vector {@code b}.
     */
    public CVector mult(CooCVector b) {
        ValidateParameters.ensureMatMultShapes(this.shape, new Shape(b.size, 1));

        Complex128[] dest = new Complex128[numRows];
        RealFieldDenseCooMatMult.standardVector(
                data, shape, b.data, b.indices, dest);

        return new CVector(dest);
    }


    /**
     * Computes the scalar multiplication of this matrix with a complex number.
     * @param b Complex valued scalar to multiply with this matrix.
     * @return The matrix-scalar product of this matrix and {@code b}.
     */
    public CMatrix mult(Complex128 b) {
        Complex128[] dest = new Complex128[data.length];
        FieldOps.scalMult(data, b, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise division between two tensors.
     *
     * @param b The denominator tensor in the element-wise quotient.
     *
     * @return The element-wise quotient of this tensor and {@code b}.
     *
     * @throws NDArrayShapeException If this tensor and {@code b}s shapes are not equal.
     */
    public CMatrix div(CMatrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseElemDiv.dispatch(shape, data, b.shape, b.data, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the scalar division of this matrix with a complex number.
     * @param b Complex valued scalar to divide this matrix by.
     * @return The matrix-scalar quotient of this matrix and {@code b}.
     */
    public CMatrix div(Complex128 b) {
        return new CMatrix(shape, Complex128Ops.scalDiv(data, b));
    }


    /**
     * <p>Computes the matrix multiplication of this matrix with itself {@code n} times. This matrix must be square.
     *
     * <p>For large {@code n} values, this method <em>may</em> be significantly more efficient than calling
     * {@link #mult(Matrix) this.mult(this)} a total of {@code n} times.
     * @param n Number of times to multiply this matrix with itself. Must be non-negative.
     * @return If {@code n=0}, then the identity
     */
    public Matrix pow(int n) {
        ValidateParameters.ensureMatMultShapes(shape, shape);
        ValidateParameters.ensureNonNegative(n);

        // Check for some quick returns.
        if (n == 0) return Matrix.I(numRows);
        if (n == 1) return copy();
        if (n == 2) return mult(this);

        Matrix result = Matrix.I(numRows);  // Start with identity matrix.
        Matrix base = this;

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
     * Computes the element-wise product of two matrices.
     * @param b The second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CMatrix elemMult(CMatrix b) {
        Complex128[] dest = new Complex128[data.length];
        RealFieldDenseElemMult.dispatch(b.data, b.shape, data, shape, dest);
        return new CMatrix(shape, dest);
    }


    /**
     * Computes the element-wise product of two matrices.
     * @param b The second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooMatrix elemMult(CooMatrix b) {
        return RealDenseSparseMatrixOps.elemMult(this, b);
    }


    /**
     * Computes the element-wise product of two matrices.
     * @param b The second matrix in the element-wise product.
     * @return The element-wise product of this matrix and {@code b}.
     */
    public CooCMatrix elemMult(CooCMatrix b) {
        return (CooCMatrix) RealFieldDenseCooMatrixOps.elemMult(this, b);
    }


    /**
     * Computes the conjugate transpose of a tensor by exchanging the first and last axes of this tensor and conjugating the
     * exchanged values.
     *
     * @return The conjugate transpose of this tensor.
     *
     * @see #H(int, int)
     * @see #H(int...)
     */
    @Override
    public Matrix H() {
        return T();
    }


    /**
     * <p>Computes the Frobenius (or <span class="latex-inline">L<sub>2, 2</sub></span>) norm of this matrix.
     *
     * <p>The Frobenius norm is defined as the square root of the sum of absolute squares of all entries in the matrix.
     *
     * @return the Frobenius of this tensor.
     */
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
    public double norm(double p) {
        return MatrixNorms.inducedNorm(this, p);
    }


    /**
     * <p>Computes the <span class="latex-inline">L<sub>p,q</sub></span> norm of this matrix.
     * <p>Some common special cases are:
     * <ul>
     *     <li>{@code p=2}, {@code q=1}: The sum of Euclidean norms of the column vectors of the matrix.</li>
     *     <li>{@code p=2}, {@code q=2}: The Frobenius norm.
     *     Equivalent to the Euclidean norm of the vector containing the singular values of the matrix.</li>
     * </ul>
     *
     * <p>The <span class="latex-inline">L<sub>p,q</sub></span> norm is computed as if by:
     * <pre>{@code
     *      double norm = 0;
     *      for(int j=0; j<src.numCols; j++) {
     *          double sum = 0;
     *          for(int i=0; i<src.numRows; i++)
     *              sum += Math.pow(Math.abs(src.get(i, j)), p);
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
     * @param object Object to check equality with this vector.
     * @return True if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * False otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        Matrix src2 = (Matrix) object;

        return RealDenseEquals.tensorEquals(data, shape, src2.data, src2.shape);
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
