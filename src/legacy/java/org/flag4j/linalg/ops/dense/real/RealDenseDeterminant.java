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

package org.flag4j.linalg.ops.dense.real;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.lu.LU;
import org.flag4j.linalg.decompositions.lu.RealLU;
import org.flag4j.util.ValidateParameters;

/**
 * This class contains methods for computing the determinant of a real dense matrix.
 */
public final class RealDenseDeterminant {

    private RealDenseDeterminant() {
        // Hide default constructor in utility class.
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static double det(Matrix A) {
        int rows = A.numRows;
        ValidateParameters.ensureSquareMatrix(rows, A.numCols);

        switch(rows) {
            case 1: // 1x1 determinant
                return A.data[0];
            case 2: // 2x2 determinant
                return A.data[0] * A.data[3] - A.data[1] * A.data[2];
            case 3: // 3x3 determinant
                double a3 = A.data[3];
                double a4 = A.data[4];
                double a5 = A.data[5];
                double a6 = A.data[6];
                double a7 = A.data[7];
                double a8 = A.data[8];

                return A.data[0]*(a4*a8 - a5*a7) - A.data[1]*(a3*a8 - a5*a6) + A.data[2]*(a3*a7 - a4*a6);
            default:
                LU<Matrix> lu = new RealLU().decompose(A);
                // Compute the determinant of P. (Check if lowest bit is zero to determine parity)
                double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;
                return detP * detTri(lu.getLU());
        }
    }


    /**
     * Computes the determinant of a square matrix using the LU factorization.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the matrix.
     * @throws IllegalArgumentException If matrix is not square.
     */
    public static double detLU(Matrix A) {
        ValidateParameters.ensureSquareMatrix(A.numRows, A.numCols);

        RealLU lu = new RealLU();
        lu.decompose(A);
        double detP = (lu.getNumRowSwaps() & 1) == 0 ? 1 : -1;

        return detP * detTri(lu.getU());
    }


    /**
     * Explicitly computes the determinant of a 3x3 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 3x3 matrix.
     */
    public static double det3(Matrix A) {
        ValidateParameters.ensureEqualShape(A.shape, new Shape(3, 3));
        double det = A.data[0] * (A.data[4] * A.data[8] - A.data[5] * A.data[7]);
        det -= A.data[1] * (A.data[3] * A.data[8] - A.data[5] * A.data[6]);
        det += A.data[2] * (A.data[3] * A.data[7] - A.data[4] * A.data[6]);
        return det;
    }


    /**
     * Explicitly computes the determinant of a 2&times;2 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 2&times;2 matrix.
     */
    public static double det2(Matrix A) {
        ValidateParameters.ensureEqualShape(A.shape, new Shape(2, 2));
        return A.data[0] * A.data[3] - A.data[1] * A.data[2];
    }


    /**
     * Explicitly computes the determinant of a 1x1 matrix.
     *
     * @param A Matrix to compute the determinant of.
     * @return The determinant of the 1x1 matrix.
     */
    public static double det1(Matrix A) {
        ValidateParameters.ensureEqualShape(A.shape, new Shape(1, 1));
        return A.data[0];
    }


    /**
     * Computes the determinant for a triangular matrix. This method does not check that the matrix is actually
     * triangular.
     *
     * @param A Triangular matrix.
     * @return The determinant of the triangular matrix.
     */
    public static double detTri(Matrix A) {
        double det = 1;
        int step = A.numCols + 1;
        int size =  A.data.length;

        // Compute the determinant of U
        for (int i=0; i<size; i += step) {
            det *= A.data[i];
        }

        return det;
    }
}
