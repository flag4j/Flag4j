/*
 * MIT License
 *
 * Copyright (c) 2023-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense.complex;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.decompositions.lu.ComplexLU;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldDeterminant;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods for computing the determinant of a complex dense matrix.
 */
public final class ComplexDenseDeterminant {

    private ComplexDenseDeterminant() {
        // Hide default constructor in utility class.
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If matrix is not square.
     */
    public static Complex128 det(CMatrix A) {
        ValidateParameters.ensureSquareMatrix(A.shape);
        Complex128 det;

        switch (A.numRows) {
            case 1:
                det = DenseFieldDeterminant.det1(A);
                break;
            case 2:
                det = DenseFieldDeterminant.det2(A);
                break;
            case 3:
                det = DenseFieldDeterminant.det3(A);
                break;
            default:
                det = detLU(A);
                break;
        }

        return det;
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws org.flag4j.util.exceptions.LinearAlgebraException If matrix is not square.
     */
    public static Complex128 detLU(CMatrix A) {
        ValidateParameters.ensureSquareMatrix(A.shape);
        LU<CMatrix> lu = new ComplexLU().decompose(A);

        double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;
        return DenseFieldDeterminant.detTriUnsafe(lu.getU()).mult(detP);
    }
}
