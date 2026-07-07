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

package org.flag4j.linalg.ops.dense.field_ops;


import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.field_arrays.AbstractDenseFieldMatrix;
import org.flag4j.arrays.dense.FieldMatrix;
import org.flag4j.linalg.decompositions.lu.FieldLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.numbers.FieldElement;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * This utility class contains methods for computing the determinant of a dense matrix whose items are elements of a
 * {@link FieldElement}.
 */
public final class DenseFieldDeterminant {

    private DenseFieldDeterminant() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the determinant of a square matrix.
     * @param mat Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static <T extends FieldElement<T>> T det(AbstractDenseFieldMatrix<?, ?, T> mat) {
        ValidateParameters.ensureSquareMatrix(mat.shape);
        T det;

        switch (mat.numRows) {
            case 1:
                det = det1(mat);
                break;
            case 2:
                det = det2(mat);
                break;
            case 3:
                det = det3(mat);
                break;
            default:
                det = detLU(mat);
                break;
        }

        return det;
    }


    /**
     * Computes the determinant of a square matrix using the {@link FieldLU LU factorization}.
     * @param mat Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static <T extends FieldElement<T>> T detLU(AbstractDenseFieldMatrix<?, ?, T> mat) {
        ValidateParameters.ensureSquareMatrix(mat.shape);
        LU<FieldMatrix<T>> lu = new FieldLU().decompose(mat);

        T detP = (lu.getNumRowSwaps() & 1) == 0 ? mat.data[0].getOne() : mat.data[0].getOne().negate();
        return detTriUnsafe(lu.getU()).mult(detP);
    }


    /**
     * Computes the determinant of a triangular matrix.
     * @param tri Triangular matrix.
     * @return The determinant of the triangular matrix {@code T}.
     */
    public static <T extends FieldElement<T>> T detTri(AbstractDenseFieldMatrix<?, ?, T> tri) {
        if(!tri.isTri()) throw new LinearAlgebraException("Matrix is not triangular.");
        return detTriUnsafe(tri);
    }


    /**
     * <p>Computes the determinant of a triangular matrix.
     * <p>WARNING: This method <em>does not</em> make <em>any</em> sanity checks. That is, no checks are made that {@code tri} is
     * square or triangular.
     * @param tri Triangular matrix. Assumed to be a square triangular matrix.
     * @return The determinant of the triangular matrix {@code tri}.
     */
    public static <T extends FieldElement<T>> T detTriUnsafe(AbstractDenseFieldMatrix<?, ?, T> tri) {
        if(tri == null || tri.data.length == 0) return null;
        T detU = tri.data[0];

        // Compute the determinant of tri
        for(int i=1, size=tri.numRows; i<size; i++)
            detU = detU.mult(tri.data[i*tri.numCols + i]);

        return detU;
    }


    /**
     * Explicitly computes the determinant of a 3x3 matrix.
     * @param mat Matrix to compute the determinant of.
     * @return The determinant of the 3x3 matrix.
     */
    public static <T extends FieldElement<T>> T det3(AbstractDenseFieldMatrix<?, ?, T> mat) {
        ValidateParameters.ensureEqualShape(mat.shape, new Shape(3, 3));
        T det = mat.data[0].mult(mat.data[4].mult(mat.data[8]).sub(mat.data[5].mult(mat.data[7])));
        det = det.sub(mat.data[1].mult(mat.data[3].mult(mat.data[8]).sub(mat.data[5].mult(mat.data[6]))));
        det = det.add(mat.data[2].mult(mat.data[3].mult(mat.data[7]).sub(mat.data[4].mult(mat.data[6]))));
        return det;
    }


    /**
     * Explicitly computes the determinant of a 2&times;2 matrix.
     * @param mat Matrix to compute the determinant of.
     * @return The determinant of the 2&times;2 matrix.
     */
    public static <T extends FieldElement<T>> T det2(AbstractDenseFieldMatrix<?, ?, T> mat) {
        ValidateParameters.ensureEqualShape(mat.shape, new Shape(2, 2));
        return mat.data[0].mult(mat.data[3]).sub(mat.data[1].mult(mat.data[2]));
    }


    /**
     * Explicitly computes the determinant of a 1x1 matrix.
     * @param mat Matrix to compute the determinant of.
     * @return The determinant of the 1x1 matrix.
     */
    public static <T extends FieldElement<T>> T det1(AbstractDenseFieldMatrix<?, ?, T> mat) {
        ValidateParameters.ensureEqualShape(mat.shape, new Shape(1, 1));
        return mat.data[0];
    }
}
