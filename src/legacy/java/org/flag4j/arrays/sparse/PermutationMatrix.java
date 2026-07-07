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

package org.flag4j.arrays.sparse;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ArrayBuilder;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.flag4j.util.exceptions.NDArrayShapeException;

import java.io.Serializable;
import java.util.Arrays;

/**
 * <p>Represents a square permutation matrix with rows and columns equal to {@link #size}, where each row and column contains exactly
 * one entry of {@code 1}, and all other entries are {@code 0}. Internally, this class stores a permutation array so that 
 * {@code permutation[i] = j} indicates that there is a {@code 1} at row {@code i}, column {@code j}.
 *
 * <p>Permutation matrices are orthogonal (and unitary in the complex case), so their inverse is equal to their
 * transpose.
 *
 * <p>Permutation matrices are useful for permuting rows or columns of another matrix.
 * <ul>
 *   <li>Left multiplying another matrix by a permutation matrix permutes the rows of that matrix.</li>
 *   <li>Right multiplying another matrix by a permutation matrix permutes the columns of that matrix.</li>
 * </ul>
 * <p>
 *
 * <p>The determinant of any permutation matrix is @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.). {@code +1} or
 * {@code -1}, depending on the parity of the permutation (i.e., the number of swaps in the matrix).
 *
 * <p>The identity matrix is a special case of a permutation matrix, corresponding to the identity permutation
 * {@code {0, 1, ..., n-1}}.
 *
 * <p>
 * <h2>Example usage:</h2>
 * <pre>{@code
 *         // Construct matrices to permute.
 *         Matrix a = new Vector(ArrayUtils.range(0, 5)).repeat(5, 1);
 *         Matrix b = a.T();
 *
 *         // Create matrix to permute rows according to (4, 2, 3, 0, 1)
 *         PermutationMatrix p1 = new PermutationMatrix(4, 2, 3, 0, 1);
 *
 *         // Permute rows of a according to (0, 1, 2, 3, 4) -> (4, 2, 3, 0, 1)
 *         Matrix aPerm = p1.leftMult(a);
 *
 *         // Permute columns of b according to (0, 1, 2, 3, 4) -> (4, 2, 3, 0, 1)
 *         Matrix bPerm = p1.T().rightMult(b);
 *
 *         // Display original matrices are their permuted counterparts.
 *         System.out.println("a:\n" + a + "\naPerm:\n" + aPerm + "\n");
 *         System.out.println("b:\n" + b + "\nbPerm:\n" + bPerm);
 * }</pre>
 */
public class PermutationMatrix implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Describes the permutation represented by this permutation matrix.
     * {@code permutation[i] = j} indicates that there is a {@code 1} at row {@code i}, column {@code j} with in permutation matrix.
     */
    protected final int[] permutation;
    /**
     * Size of this permutation matrix.
     */
    public final int size;
    /**
     * Shape of this permutation matrix.
     */
    public final Shape shape;


    /**
     * Creates a permutation matrix which is equivalent to the identity matrix of the specified size.
     * @param size Size of the permutation matrix. That is, the number of rows and columns
     */
    public PermutationMatrix(int size) {
        this.size = size;
        shape = new Shape(size, size);
        permutation = ArrayBuilder.intRange(0, size);
    }


    /**
     * Creates a permutation matrix which is equivalent to the identity matrix of the specified size.
     * @param shape Shape of the permutation matrix. That is, the number of rows and columns. Must be a square shape.
     * @throws LinearAlgebraException If {@code shape} is not square.
     */
    public PermutationMatrix(Shape shape) {
        ValidateParameters.ensureSquareMatrix(shape);
        this.shape = shape;
        this.size = shape.getSize(0);
        permutation = ArrayBuilder.intRange(0, size);
    }


    /**
     * Copy constructor which creates a deep copy of the {@code src} permutation matrix.
     * @param src The permutation matrix to copy.
     */
    public PermutationMatrix(PermutationMatrix src) {
        this.size = src.size;
        this.shape = src.shape;
        this.permutation = src.permutation.clone();
    }


    /**
     * <p>Constructs a permutation matrix from the specified {@code permutation}.
     *
     * <p>This constructor will explicitly verify that {@code permutation} is a valid permutation. It is <em>highly</em> recommended
     * to do this. However, there is a
     *
     * @param permutation Array specifying the permutation. Must contain a permutation of {@code {0, 1, ..., permutation.length-1}}.
     * {@code permutation[i] = j} indicates that there is a {@code 1} at row {@code i}, column {@code j}.
     * @throws IllegalArgumentException {@code permutation} is <em>not</em> a permutation of
     * {@code {0, 1, ..., permutation.length-1}.
     */
    public PermutationMatrix(int... permutation) {
        this(permutation, true);
    }


    /**
     * <p>Constructs a permutation matrix from the specified {@code permutation}. This constructor also accepts a flag indicating if an
     * explicit check should be made to enforce that the {@code permutation} array is a valid permutation.
     *
     * <p> It is <em>highly</em> recommended to use {@link #PermutationMatrix(int[])} or set {@code ensurePermutation = true}. However,
     * if there is absolute confidence in the validity of the {@code permutation} array, then setting
     * {@code ensurePermutation = false} <em>may</em> yield very slight performance benefits.
     *
     * @param permutation Array specifying the permutation. Must contain a permutation of {@code {0, 1, ..., permutation.length-1}}.
     * {@code permutation[i] = j} indicates that there is a {@code 1} at row {@code i}, column {@code j}.
     * @param ensurePermutation Flag indicating if an explicit check should be made to verify that {@code permutation} is a valid
     * permutation of {@code {0, 1, ..., permutation.length-1}}.
     * <ul>
     *     <li>If {@code true}: an explicit check will be made that {@code permutation} is a valid permutation.</li>
     *     <li>If {@code false}: <em>NO</em> check will be made to ensure {@code permutation} is a valid permutation.</li>
     * </ul>
     *
     * @throws IllegalArgumentException If {@code ensurePermutation == true} and {@code permutation} is <em>not</em> a permutation of
     * {@code {0, 1, ..., permutation.length-1}.
     */
    public PermutationMatrix(int[] permutation, boolean ensurePermutation) {
        if(ensurePermutation) ValidateParameters.ensurePermutation(permutation);
        this.size = permutation.length;
        this.shape = new Shape(size, size);
        this.permutation = permutation.clone();
    }


    /**
     * Returns the permutation represented by this permutation matrix.
     * @return The permutation represented by this permutation matrix.
     */
    public int[] getPermutation() {
        return permutation;
    }


    /**
     * Creates a permutation matrix with the specified column permutation. That is, a permutation matrix such that
     * right multiplying it with another matrix results in permuting th columns of that matrix according to {@code colPermutation}.
     *
     * @param colPermutation Array specifying column permutation. The entry {@code x} at index {@code i} indicates that column
     * {@code i} has been swapped with column {@code x}.
     * {@link ValidateParameters#ensurePermutation(int...) permutation array}.
     * @return A permutation matrix that when right multiplied to a matrix results in permuting the columns of that matrix according
     * to {@code colPermutation}.
     * @throws IllegalArgumentException If {@code colPermutation} is not a valid permutation.
     */
    public static PermutationMatrix fromColSwaps(int[] colPermutation) {
        int[] rowPerm = new int[colPermutation.length];

        for (int i=0; i<colPermutation.length; i++)
            rowPerm[colPermutation[i]] = i;

        return new PermutationMatrix(rowPerm);
    }


    /**
     * Creates a copy of this permutation matrix.
     * @return A copy of this permutation matrix.
     */
    public PermutationMatrix copy() {
        return new PermutationMatrix(this);
    }


    /**
     * Checks if this permutation matrix is equal to the given object. A permutation matrix is considered equal to an
     * object if that object is also a permutation matrix and represents the same matrix.
     * @param object Object to compare to this permutation matrix.
     * @return True if {@code b} is a permutation matrix and equivalent to this matrix (in terms of matrix equality).
     */
    @Override
    public boolean equals(Object object) {
        // Check for quick returns.
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        PermutationMatrix src2 = (PermutationMatrix) object;

        return Arrays.equals(permutation, src2.permutation);
    }


    /**
     * Returns a hashcode for this permutation matrix by calling {@link Arrays#hashCode(int[]) Arrays.hashCode(permutation)}.
     * @return The hashcode for this permutation matrix.
     */
    @Override
    public int hashCode() {
        return Arrays.hashCode(permutation);
    }


    /**
     * Computes the matrix-matrix multiplication between two permutation matrices.
     * @param b The matrix to multiply to this permutation matrix.
     * @return The matrix=matrix product of this permutation matrix with {@code b}.
     * @throws NDArrayShapeException If {@code this.size != b.size}.
     */
    public PermutationMatrix mult(PermutationMatrix b) {
        if(this.size != b.size) {
            throw new NDArrayShapeException("Shapes not compatible with matrix multiplication: "
                    + shape + " and " + b.shape + ".");
        }

        int[] prod = new int[size];
        for(int i=0; i<size; i++)
            prod[i] = b.permutation[permutation[i]];

        return new PermutationMatrix(prod);
    }


    /**
     * Left multiplies this permutation matrix to the specified matrix. This will have the effect of swapping rows in
     * the src matrix.
     * @param src The matrix to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} matrix.
     * @see #rightMult(Matrix)
     * @throws IllegalArgumentException If the number of rows in {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Matrix leftMult(Matrix src) {
        ValidateParameters.ensureAllEqual(size, src.numRows);
        double[] destEntries = new double[src.data.length];
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = permutation[rowIdx];
            System.arraycopy(src.data, colIdx*src.numCols, destEntries, rowIdx*src.numCols, src.numCols);
        }

        return new Matrix(src.shape, destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified vector. This will have the effect of swapping rows in
     * the src vector. The vector will be treated as a column vector.
     * @param src The vector to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} vector.
     * @see #rightMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Vector leftMult(Vector src) {
        ValidateParameters.ensureAllEqual(size, src.size);
        double[] destEntries = new double[src.data.length];
        double[] srcData = src.data;

        for(int rowIdx=0; rowIdx<size; rowIdx++)
            destEntries[rowIdx] = srcData[permutation[rowIdx]];

        return new Vector(destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified matrix. This will have the effect of swapping rows in
     * the src matrix.
     * @param src The matrix to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} matrix.
     * @see #rightMult(CMatrix)
     * @throws IllegalArgumentException If the number of rows in {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CMatrix leftMult(CMatrix src) {
        ValidateParameters.ensureAllEqual(size, src.numRows);
        Complex128[] destEntries = new Complex128[src.data.length];
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = permutation[rowIdx];
            System.arraycopy(src.data, colIdx*src.numCols, destEntries, rowIdx*src.numCols, src.numCols);
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Left multiplies this permutation matrix to the specified vector. This will have the effect of swapping rows in
     * the src vector. The vector will be treated as a column vector.
     * @param src The vector to left multiply this permutation matrix to.
     * @return The result of left multiplying this permutation matrix to the {@code src} vector.
     * @see #rightMult(CVector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CVector leftMult(CVector src) {
        ValidateParameters.ensureAllEqual(size, src.size);
        Complex128[] destEntries = new Complex128[src.data.length];

        for(int rowIdx=0; rowIdx<size; rowIdx++)
            destEntries[rowIdx] = src.data[permutation[rowIdx]];

        return new CVector(destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified matrix. This is equivalent to swapping columns in the
     * {@code src} matrix.
     * @param src The matrix to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} matrix.
     * @throws IllegalArgumentException If the number of columns in {@code src} does not match the size of this
     * permutation matrix.
     * @see #leftMult(Matrix)
     */
    public Matrix rightMult(Matrix src) {
        ValidateParameters.ensureAllEqual(size, src.numCols);
        double[] destEntries = new double[src.data.length];

        int colIdx;
        int rowOffset;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = permutation[rowIdx];

            for(int j=0; j<src.numRows; j++) {
                rowOffset = j*src.numCols;
                destEntries[rowOffset + colIdx] = src.data[rowOffset + rowIdx];
            }
        }

        return new Matrix(src.shape, destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified vector. This will have the effect of swapping columns in
     * the src vector. The vector will be treated as a row vector.
     * @param src The vector to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} vector.
     * @see #leftMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public Vector rightMult(Vector src) {
        // For vectors, left/right multiplication is equivalent since vectors do not have orientation
        // (i.e., row/column vectors.)
        return leftMult(src);
    }


    /**
     * Right multiplies this permutation matrix to the specified matrix. This is equivalent to swapping columns in the
     * {@code src} matrix.
     * @param src The matrix to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} matrix.
     * @throws IllegalArgumentException If the number of columns in {@code src} does not match the size of this
     * permutation matrix.
     * @see #leftMult(Matrix)
     */
    public CMatrix rightMult(CMatrix src) {
        ValidateParameters.ensureAllEqual(size, src.numCols);
        Complex128[] destEntries = new Complex128[src.data.length];
        final int rows = src.numRows;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            int colIdx = permutation[rowIdx];

            for(int j=0; j<src.numRows; j++) {
                int rowOffset = j*src.numCols;
                destEntries[rowOffset + colIdx] = src.data[rowOffset + rowIdx];
            }
        }

        return new CMatrix(src.shape, destEntries);
    }


    /**
     * Right multiplies this permutation matrix to the specified vector. This will have the effect of swapping columns in
     * the src vector. The vector will be treated as a row vector.
     * @param src The vector to right multiply this permutation matrix to.
     * @return The result of right multiplying this permutation matrix to the {@code src} vector.
     * @see #leftMult(Vector)
     * @throws IllegalArgumentException If size of {@code src} does not equal the size of this permutation
     * matrix.
     */
    public CVector rightMult(CVector src) {
        // For vectors, left/right multiplication is equivalent since vectors do not have orientation
        // (i.e., row/column vectors.)
        return leftMult(src);
    }


    /**
     * Swaps two rows in this permutation matrix.
     * @param row1 First row to swap in the permutation matrix.
     * @param row2 Second row to swap in the permutation matrix.
     * @throws ArrayIndexOutOfBoundsException If either {@code row1} or {@code row2} is out of bounds of this permutation
     * matrix.
     */
    public void swapRows(int row1, int row2) {
        ArrayUtils.swap(permutation, row1, row2);
    }


    /**
     * Swaps two columns in this permutation matrix.
     * @param col1 First column to swap in the permutation matrix.
     * @param col2 Second column to swap in the permutation matrix.
     * @throws ArrayIndexOutOfBoundsException If either {@code col1} or {@code col2} is out of bounds of this permutation
     * matrix.
     */
    public void swapCols(int col1, int col2) {
        ValidateParameters.validateArrayIndices(size, col1, col2);
        // Find locations of items with the given columns.
        int idx1 = ArrayUtils.indexOf(permutation, col1);
        int idx2 = ArrayUtils.indexOf(permutation, col2);
        ArrayUtils.swap(permutation, idx1, idx2); // Swap values.
    }


    /**
     * Permutes rows of this permutation matrix.
     * @param swaps Defines row swaps of this permutation matrix. The entry {@code x} at index {@code i}
     *              represents row {@code i} has been swapped with row {@code x}. This must be a
     *              {@link ValidateParameters#ensurePermutation(int...)  permutation} array.
     * @throws IllegalArgumentException If {@code swaps} is not the same length as the number of rows/columns in this
     * permutation matrix. Or, if {@code swaps} is not a
     * {@link ValidateParameters#ensurePermutation(int...)  permutation} array.
     */
    public void permuteRows(int[] swaps) {
        ValidateParameters.ensurePermutation(swaps);
        ValidateParameters.ensureArrayLengthsEq(swaps.length, permutation.length);
        System.arraycopy(swaps, 0, permutation, 0, swaps.length);
    }


    /**
     * Computes the inverse/transpose of this permutation matrix.
     * @return The inverse/transpose of this permutation matrix.
     */
    public PermutationMatrix inv() {
        return T();
    }


    /**
     * Computes the transpose/inverse of this permutation matrix.
     * @return The transpose/inverse of this permutation matrix.
     */
    public PermutationMatrix T() {
        int[] transpose = new int[size];

        for(int i=0; i<size; i++)
            transpose[permutation[i]] = i;

        return new PermutationMatrix(transpose);
    }


    /**
     * Computes the trace of this permutation matrix.
     * @return The trace of this permutation matrix.
     */
    public int trace() {
        return tr();
    }


    /**
     * Computes the trace of this permutation matrix. Alias for {@link #trace()}.
     * @return The trace of this permutation matrix.
     */
    public int tr() {
        int trace = 0;

        for(int i=0; i<size; i++)
            if(permutation[i]==i) trace += 1;

        return trace;
    }


    /**
     * Computes the number of row/column swaps required for this permutation matrix to be converted to the identity matrix.
     * @return The total number of row/column swaps required to convert this permutation matrix to the identity matrix.
     */
    public int computeSwaps() {
        boolean[] visited = new boolean[permutation.length];
        int totalSwaps = 0;

        for (int i = 0; i < permutation.length; i++) {
            if (!visited[i]) {
                visited[i] = true;

                if (permutation[i] != i) {
                    int cycleSize = 1;
                    int next = permutation[i];

                    while (next != i) {
                        visited[next] = true;
                        next = permutation[next];
                        cycleSize++;
                    }

                    totalSwaps += cycleSize - 1;
                }
            }
        }

        return totalSwaps;
    }


    /**
     * Computes the determinant of this permutation matrix (will be +/- 1).
     * @return The determinant of this permutation matrix (will be +/- 1).
     */
    public int det() {
        // Determinant is 1 for an even number of swaps, and -1 for an odd number of swaps.
        return (computeSwaps() & 1) == 0 ? 1 : -1;
    }


    /**
     * Converts this permutation matrix to a {@link Matrix real dense matrix}.
     * @return A real dense matrix which is equivalent to this permutation matrix.
     * @see #toCoo()
     * @see #toCsr()
     */
    public Matrix toDense() {
        // TODO: Once integer matrices are implemented, return that instead of a double matrix.
        double[] data = new double[size*size];
        int rowOffset = 0;
        int colIdx;

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            colIdx = permutation[rowIdx];
            data[rowOffset + colIdx] = 1.0;
            rowOffset += size;
        }

        return new Matrix(shape, data);
    }


    /**
     * Converts this permutation matrix to a {@link CooMatrix real sparse COO matrix}.
     * @return real sparse COO matrix which is equivalent to this permutation matrix.
     *
     * @see #toDense()
     * @see #toCsr()
     */
    public CooMatrix toCoo() {
        // TODO: Once integer matrices are implemented, return that instead of a double matrix.
        double[] data = new double[size];
        int[] rowIndices = new int[size];
        int[] colIndices = new int[size];

        for(int rowIdx=0; rowIdx<size; rowIdx++) {
            data[rowIdx] = 1.0;
            rowIndices[rowIdx] = rowIdx;
            colIndices[rowIdx] = permutation[rowIdx];
        }

        return CooMatrix.unsafeMake(shape, data, rowIndices, colIndices);
    }


    /**
     * Converts this permutation matrix to a {@link CooMatrix real sparse COO matrix}.
     * @return real sparse COO matrix which is equivalent to this permutation matrix
     *
     * @see #toDense()
     * @see #toCoo()
     */
    public CsrMatrix toCsr() {
        // TODO: Once integer matrices are implemented, return that instead of a double matrix.
        return toCoo().toCsr();
    }


    /**
     * Converts this permutation matrix to a human-readable string.
     * @return This permutation matrix represented as a human-readable string.
     */
    public String toString() {
        return "shape=" + new Shape(size, size) + "\n" +
                "permutation: " + Arrays.toString(permutation);
    }
}
