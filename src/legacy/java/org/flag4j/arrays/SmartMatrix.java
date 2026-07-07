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

package org.flag4j.arrays;

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.smart_visitors.*;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.SemiringMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.FieldElement;
import org.flag4j.numbers.SemiringElement;


/**
 * <p>The {@code SmartMatrix} class provides a unified interface for performing operations on matrices of various types
 * without requiring users to know the specific underlying implementation. It wraps a base matrix type, such as
 * {@link Matrix}, {@link CMatrix}, {@link FieldMatrix}, or {@link SemiringMatrix}, and delegates operations to the
 * appropriate concrete implementation. The concrete implementation must implement {@link MatrixMixin}.
 *
 * <h2>Features:</h2>
 * <p>The {@code SmartMatrix} class supports most basic matrix operations but may be more limited than the base concrete matrix types.
 * <ul>
 *   <li>Support for element-wise operations such as addition, subtraction, multiplication, and division.</li>
 *   <li>Matrix-specific operations, including transpose and conjugate transpose.</li>
 *   <li>Compatibility with heterogeneous matrix types, allowing flexible operation dispatch.</li>
 *   <li>Support for methods which return a scalar (e.g. {@link #tr(Class)}).</li>
 *   <li>Human-readable string representations and standard equality checks.</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 *
 * <ul>
 *     <li>Creating and adding a real dense and complex dense matrix:
 * <pre>{@code
 * double[][] realData = {{1.0, 2.0}, {3.0, 4.0}};
 * SmartMatrix realMatrix = new SmartMatrix(realData);
 *
 * Complex128[][] complexData = {
 *     {new Complex128(1.0, 1.0), new Complex128(2.0, 2.0)},
 *     {new Complex128(3.0, 3.0), new Complex128(4.0, 4.0)}
 * };
 * SmartMatrix complexMatrix = new SmartMatrix(complexData);
 *
 * // Compute real-complex element-wise sum.
 * SmartMatrix result = realMatrix.add(complexMatrix);
 * }</pre></li>
 *
 * <li>Creating and multiplying a real dense and complex sparse CSR matrix:
 * <pre>{@code
 * // Create real dense matrix.
 * SmartMatrix realDenseMatrix = new SmartMatrix(new Matrix(...));
 *
 * // Create complex sparse CSR matrix.
 * SmartMatrix complexCsrMatrix = new SmartMatrix(new CsrCMatrix(...));
 *
 * // Compute real-dense complex-csr matrix multiplication.
 * SmartMatrix result = realDenseMatrix.mult(complexCsrMatrix);
 * }</pre></li>
 *
 * <li>Example of attempting to compute an unsupported operation:
 * <pre>{@code
 * // Create real dense matrix.
 * SmartMatrix realDenseMatrix = new SmartMatrix(new Matrix(...));
 *
 * // Create real sparse COO matrix.
 * SmartMatrix complexCsrMatrix = new SmartMatrix(new CooMatrix(...));
 *
 * // Will throw UnsupportedOperationException as element-wise division
 * //   is not supported between dense and sparse matrices.
 * SmartMatrix result = realDenseMatrix.div(complexCsrMatrix);
 * }</pre></li>
 *
 * <li>Specifying class for methods which return a scalar:
 * <pre>{@code
 * // Create real dense matrix.
 * SmartMatrix realMatrix = new SmartMatrix(new Matrix(...));
 *
 * // Class of the expected return type must be provided.
 * // If class is incorrect, a ClassCastException will be thrown.
 * Double traceReal = realMatrix.tr(Double.class);
 *
 * // Create complex dense matrix.
 * SmartMatrix complexMatrix = new SmartMatrix(new CMatrix(...));
 * Complex128 traceComplex = complexMatrix.tr(Complex128.class);
 * }</pre></li>
 * </ul>
 *
 * <h2>Notes:</h2>
 * <ul>
 *   <li>Operations between incompatible base matrix types will result in an {@link UnsupportedOperationException}.</li>
 *   <ul><li>For example: attempting to divide a dense matrix by a sparse matrix or adding a {@link Matrix real matrix}
 *   to a general {@link FieldMatrix field matrix}.</li></ul>
 *   <li>Equality checks (by {@link #equals(Object)}) are strict and include type
 *   comparison as well as shape and numerical equivalence. This means even if a {@link Matrix} and {@link CMatrix} are numerically
 *   equal, the {@link #equals(Object)} method will return {@code false} because they are not the same type.
 *   </li>
 * </ul>
 *
 * @see MatrixMixin
 * @see Matrix
 * @see CMatrix
 * @see FieldMatrix
 * @see SemiringMatrix
 */
public class SmartMatrix {

    // TODO: Implement SmartTensor and SmartVector.
    // TODO: Investigate other methods that SmartMatrix should implement.

    /**
     * The matrix which backs this {@code SmartMatrix} instance.
     */
    private final MatrixMixin<?, ?, ?, ?> matrix;


    /**
     * Constructs a {@code SmartMatrix} which is backed by the specified matrix.
     *
     * @param matrix The backing matrix for this {@code SmartMatrix}.
     */
    public SmartMatrix(MatrixMixin<?, ?, ?, ?> matrix) {
        this.matrix = matrix;
    }


    /**
     * Constructs a {@code SmartMatrix} from a 2D primitive double array. The matrix which backs this {@code SmartMatrix} will be an
     * instance of {@link Matrix}.
     *
     * @param data Array specifying shape and items of this {@code SmartMatrix}.
     */
    public SmartMatrix(double[][] data) {
        matrix = new Matrix(data);
    }


    /**
     * Constructs a {@code SmartMatrix} from a 2D {@link Complex128} array. The matrix which backs this {@code SmartMatrix} will be an
     * instance of {@link CMatrix}.
     *
     * @param data Array specifying shape and items of this {@code SmartMatrix}.
     */
    public SmartMatrix(Complex128[][] matrix) {
        this.matrix = new CMatrix(matrix);
    }


    /**
     * Constructs a {@code SmartMatrix} from a 2D {@link FieldElement} array. The matrix which backs this {@code SmartMatrix} will be an
     * instance of {@link FieldMatrix}.
     *
     * @param data Array specifying shape and items of this {@code SmartMatrix}.
     */
    public <T extends FieldElement<T>> SmartMatrix(T[][] matrix) {
        this.matrix = new FieldMatrix<>(matrix);
    }


    /**
     * Constructs a {@code SmartMatrix} from a 2D {@link SemiringMatrix} array. The matrix which backs this {@code SmartMatrix} will be an
     * instance of {@link SemiringMatrix}.
     *
     * @param data Array specifying shape and items of this {@code SmartMatrix}.
     */
    public <T extends SemiringElement<T>> SmartMatrix(T[][] matrix) {
        this.matrix = new SemiringMatrix<>(matrix);
    }


    /**
     * Gets reference to the matrix which backs this {@code SmartMatrix}.
     *
     * @return The matrix which backs this {@code SmartMatrix}.
     */
    public MatrixMixin<?, ?, ?, ?> getMatrix() {
        return matrix;
    }


    /**
     * Computes the element-wise sum of two matrices.
     *
     * @param b The second matrix in the element-wise sum.
     *
     * @return The element-wise sum of this matrix and {@code b}.
     */
    public SmartMatrix add(SmartMatrix b) {
        var visitor = new AddVisitor(b.matrix);
        return new SmartMatrix(matrix.accept(visitor));
    }


    /**
     * Computes the element-wise difference of two matrices.
     *
     * @param b The second matrix in the element-wise difference.
     *
     * @return The element-wise difference of this matrix and {@code b}.
     */
    public SmartMatrix sub(SmartMatrix b) {
        var visitor = new SubVisitor(b.matrix);
        return new SmartMatrix(matrix.accept(visitor));
    }


    /**
     * Computes the element-wise product of two matrices.
     *
     * @param b The second matrix in the element-wise product.
     *
     * @return The element-wise product of this matrix and {@code b}.
     */
    public SmartMatrix elemMult(SmartMatrix b) {
        var visitor = new ElemMultVisitor(b.matrix);
        return new SmartMatrix(matrix.accept(visitor));
    }


    /**
     * Computes the element-wise quotient of two matrices.
     *
     * @param b The second matrix in the element-wise quotient.
     *
     * @return The element-wise quotient of this matrix and {@code b}.
     */
    public SmartMatrix div(SmartMatrix b) {
        var visitor = new DivVisitor(b.matrix);
        return new SmartMatrix(matrix.accept(visitor));
    }


    /**
     * Computes the matrix multiplication of two matrices.
     *
     * @param b The second matrix in the matrix multiplication problem.
     *
     * @return The result of the matrix multiplication of this matrix and {@code b}.
     */
    public SmartMatrix mult(SmartMatrix b) {
        var visitor = new MatMultVisitor(b.matrix);
        return new SmartMatrix(matrix.accept(visitor));
    }


    /**
     * Computes the transpose of this matrix.
     *
     * @return The transpose of this matrix.
     */
    public SmartMatrix T() {
        return new SmartMatrix(matrix.T());
    }


    /**
     * Computes the conjugate transpose of this matrix. This may not be supported for all matrix types.
     *
     * @return The conjugate transpose of this matrix.
     */
    public SmartMatrix H() {
        return new SmartMatrix(matrix.H());
    }


    /**
     * <p>Computes the trace of this matrix. The result is a scalar value whose type is dependent on the underlying backing
     * matrix of this {@code SmartMatrix} instance (e.g., {@link Double}, {@link Complex128}, etc.).
     *
     * <p>This method uses a {@link Class} object to specify the expected type of the resulting scalar.
     * It ensures type safety by attempting to cast the result to the specified type. If the cast is
     * not valid, a {@link ClassCastException} will be thrown at runtime.
     *
     * <p><h2>Usage Examples:</h2>
     * <ul>
     *     <li> Real Matrix:
     * <pre>{@code
     * SmartMatrix realMatrix = new SmartMatrix(new double[][] {
     *     {1.0, 2.0},
     *     {3.0, 4.0}});
     * Double128 realTrace = realMatrix.tr(Double.class);
     * System.out.println("Trace: " + realMatrix);}</pre>
     *     </li>
     *     <li> Complex Matrix:
     *<pre>{@code
     * SmartMatrix complexMatrix = new SmartMatrix(new Complex128[][] {
     *     {new Complex128(1.0, 1.0), new Complex128(2.0, 2.0)},
     *     {new Complex128(3.0, 3.0), new Complex128(4.0, 4.0)}});
     * Complex128 complexTrace = complexMatrix.tr(Complex128.class);
     * System.out.println("Trace: " + complexTrace);}</pre>
     *     </li>
     * </ul>
     *
     * @param <T> The type of the scalar expected as the result.
     * @param type The {@link Class} object representing the expected type of the resulting scalar.
     * @return The trace of this matrix as an instance of the specified type {@code T}.
     * @throws ClassCastException If the resulting trace cannot be cast to the specified type {@code T}.
     * @throws UnsupportedOperationException If the trace computation is not supported by the backing matrix of this {@code
     * SmartMatrix} instance.
     */
    public <T> T tr(Class<T> type) {
        return type.cast(matrix.tr());
    }


    /**
     * Gets the element from this matrix at the specified indices. The result is a scalar value whose type is
     * dependent on the underlying backing matrix of
     * this {@code SmartMatrix} instance (e.g., {@link Double}, {@link Complex128}, etc.).
     *
     * <p>This method uses a {@link Class} object to specify the expected type of the resulting scalar.
     * It ensures type safety by attempting to cast the result to the specified type. If the cast is
     * not valid, a {@link ClassCastException} will be thrown at runtime.
     *
     * @param rowIdx Row index of the element to get.
     * @param colIdx Column index of the element to get.
     * @param type The {@link Class} object representing the expected type of the resulting scalar.
     * @return The element of this matrix at the specified row and column index.
     * @throws ClassCastException If the resulting trace cannot be cast to the specified type {@code T}.
     * @throws UnsupportedOperationException If the trace computation is not supported by the backing matrix of this {@code
     * @param <T> The type of the scalar expected as the result.
     */
    public <T> T get(int rowIdx, int colIdx, Class<T> type) {
        return type.cast(matrix.get(rowIdx, colIdx));
    }


    /**
     * Gets the shape of this matrix.
     *
     * @return The shape of this matrix.
     */
    public Shape getShape() {
        return matrix.getShape();
    }


    /**
     * Checks if an object is equal to this matrix object.
     *
     * @param object Object to check equality with this matrix.
     *
     * @return {@code true} if the two matrices have the same shape, are numerically equivalent, and are of type {@link Matrix}.
     * {@code false} otherwise.
     */
    @Override
    public boolean equals(Object object) {
        if(object == null) return false;
        if(object == this) return true;
        if(getClass() != object.getClass()) return false;

        return matrix.equals(object);
    }


    @Override
    public int hashCode() {
        int hash = 17;
        hash = hash*31 + getClass().hashCode();
        hash = hash*31 + (matrix == null ? 0 : matrix.hashCode());
        return hash;
    }


    /**
     * Converts this matrix to a human-readable string.
     *
     * @return A human-readable string representation of this matrix.
     */
    public String toString() {
        return "SmartMatrix type: " + this.matrix.getClass().getSimpleName() + "\n" + matrix.toString();
    }
}
