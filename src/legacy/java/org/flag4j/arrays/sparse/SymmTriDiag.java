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
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.util.ValidateParameters;

import java.io.Serializable;
import java.util.Arrays;


/**
 * <p>A real symmetric tri-diagonal matrix. This class stores the non-zero values of the symmetric tri-diagonal matrix
 * and has limited support for ops with such a matrix.
 *
 * <p>A matrix is symmetric tri-diagonal if it is symmetric and all values below the first sub-diagonal and above the first
 * super-diagonal are zero.
 *
 * <p>For example, the following matrix is in symmetric tri-diagonal form where each {@code X} may hold a different value (provided
 * the matrix is symmetric):
 * <pre>
 *     [[ X X 0 0 0 ]
 *      [ X X X 0 0 ]
 *      [ 0 X X X 0 ]
 *      [ 0 0 X X X ]
 *      [ 0 0 0 X X ]]</pre>
 * 
 */
public class SymmTriDiag implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Stores the diagonal items of this symmetric tri-diagonal matrix.
     */
    final double[] diag;
    /**
     * Stores the first sub/super diagonal items of this symmetric tri-diagonal matrix.
     */
    final double[] offDiag;
    /**
     * The size (i.e., number of rows and columns) of this symmetric tri-diagonal matrix.
     */
    final int size;


    /**
     * Constructs a symmetric tri-diagonal matrix with the provided diagonal and off-diagonal items.
     * @param diag Diagonal items of the symmetric tri-diagonal matrix.
     * @param offDiag Sub/super diagonal items of the symmetric tri-diagonal matrix.
     */
    public SymmTriDiag(double[] diag, double[] offDiag) {
        ValidateParameters.ensureArrayLengthsEq(diag.length-1, offDiag.length);

        this.size = diag.length;
        this.diag = diag;
        this.offDiag = offDiag;
    }


    /**
     * Gets the shape of this symmetric tri-diagonal matrix.
     * @return The shape of this symmetric tri-diagonal matrix.
     */
    public Shape getShape() {
        return new Shape(size, size);
    }


    /**
     * Gets the entry within this symmetric tri-diagonal matrix at the specified indices.
     * @param rowIdx Row index of entry to get.
     * @param colIdx Column index of entry to get.
     * @return The entry of this symmetric tri-diagonal matrix at the specified indices.
     * @throws IndexOutOfBoundsException If either index is out of bounds for this matrix.
     */
    public double get(int rowIdx, int colIdx) {
        // Ensure indices are in range for this matrix.
        ValidateParameters.validateArrayIndices(size, rowIdx, colIdx);
        int absDiff = Math.abs(rowIdx-colIdx);

        if(absDiff == 0) {
            return diag[rowIdx]; // Diagonal value
        } else if(absDiff == 1) {
            return offDiag[Math.min(rowIdx, colIdx)]; // on sub/super-diagonal
        } else {
            // Not within principle tri-diagonal.
            return 0.0;
        }
    }


    /**
     * Converts this symmetric tri-diagonal matrix to an equivalent dense matrix.
     * @return A dense matrix equivalent to this symmetric tri-diagonal matrix.
     */
    public Matrix toDense() {
        double[] entries = new double[size*size];

        // Set the diagonal and off-diagonal values.
        for(int i=0; i<offDiag.length; i++) {
            int rowOffset = i*size + i;
            entries[rowOffset] = diag[i];
            entries[rowOffset + 1] = offDiag[i];
            entries[rowOffset + size] = offDiag[i];
        }

        entries[entries.length-1] = diag[diag.length-1];

        return new Matrix(size, size, entries);
    }


    /**
     * Checks if an object is equal to this symmetric tri-diagonal matrix. An object is considered equal to this matrix if it is
     * an instance of {@link org.flag4j.arrays.sparse.SymmTriDiag} and all diagonal and off diagonal items are equal.
     * @param object Object to compare to this symmetric tri-diagonal matrix.
     * @return True if {@code b} is an instance of {@link org.flag4j.arrays.sparse.SymmTriDiag} and all diagonal and off diagonal items are equal to the
     * corresponding values in this symmetric tri-diagonal matrix.
     */
    public boolean equals(Object object) {
        // Check for quick returns.
        if(this == object) return true;
        if(object == null || object.getClass() != getClass()) return false;

        SymmTriDiag src2 = (SymmTriDiag) object;

        return Arrays.equals(src2.diag, diag) && Arrays.equals(src2.offDiag, offDiag);
    }


    /**
     * Creates a hash code for this symmetric tri-diagonal matrix.
     * @return An integer hash code for this symmetric tri-diagonal matrix.
     */
    @Override
    public int hashCode() {
        int hash = 17 + Arrays.hashCode(diag);
        hash = hash*31 + Arrays.hashCode(offDiag);
        return hash;
    }
}
