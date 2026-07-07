/*
 * MIT License
 *
 * Copyright (c) 2022-2025. Jacob Watters
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

package org.flag4j.linalg;


import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.decompositions.svd.ComplexSVD;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.decompositions.svd.SVD;
import org.flag4j.linalg.solvers.lstsq.ComplexLstsqSolver;
import org.flag4j.linalg.solvers.lstsq.RealLstsqSolver;
import org.flag4j.util.Flag4jConstants;

/**
 * This class contains several methods for computing the subspace of a matrix.
 */
public final class SubSpace {

    private SubSpace() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes an orthonormal basis of the column space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the column space.
     * @return A matrix containing as its columns, an orthonormal basis for the column space of the {@code src} matrix.
     */
    public static Matrix getColSpace(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        int rank = svd.getRank();
        return svd.getU().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the row space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the row space.
     * @return A matrix containing as its columns, an orthonormal basis for the row space of the {@code src} matrix.
     */
    public static Matrix getRowSpace(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        int rank = svd.getRank();
        return svd.getV().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the null space.
     * @return A matrix containing as its columns, an orthonormal basis for the null space of the {@code src} matrix.
     */
    public static Matrix getNullSpace(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        int rank = svd.getRank();
        int numCols = Math.min(src.numRows, src.numCols);

        return src.numCols-rank==0 ?
                new Matrix(src.numCols, 1) :
                svd.getV().getSlice(0, src.numCols, rank, numCols);
    }


    /**
     * Computes an orthonormal basis of the left null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the left null space.
     * @return A matrix containing as its columns, an orthonormal basis for the left null space of the {@code src} matrix.
     */
    public static Matrix getLeftNullSpace(Matrix src) {
        SVD<Matrix> svd = new RealSVD().decompose(src);
        int rank = svd.getRank();
        return src.numRows-rank==0 ?
                new Matrix(src.numCols, 1) :
                svd.getU().getSlice(0, src.numRows, rank, src.numRows);
    }


    /**
     * Computes an orthonormal basis of the column space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the column space.
     * @return A matrix containing as its columns, an orthonormal basis for the column space of the {@code src} matrix.
     */
    public static CMatrix getColSpace(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        int rank = svd.getRank();
        return svd.getU().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the row space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the row space.
     * @return A matrix containing as its columns, an orthonormal basis for the row space of the {@code src} matrix.
     */
    public static CMatrix getRowSpace(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        int rank = svd.getRank();
        return svd.getV().getSlice(0, src.numRows, 0, rank);
    }


    /**
     * Computes an orthonormal basis of the null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the null space.
     * @return A matrix containing as its columns, an orthonormal basis for the null space of the {@code src} matrix.
     */
    public static CMatrix getNullSpace(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        int rank = svd.getRank();
        int numCols = Math.min(src.numRows, src.numCols);

        return src.numCols-rank==0 ?
                new CMatrix(src.numCols, 1) :
                svd.getV().getSlice(0, numCols, rank, numCols);
    }


    /**
     * Computes an orthonormal basis of the left null space of a specified matrix.
     * @param src Matrix to compute orthonormal basis of the left null space.
     * @return A matrix containing as its columns, an orthonormal basis for the left null space of the {@code src} matrix.
     */
    public static CMatrix getLeftNullSpace(CMatrix src) {
        SVD<CMatrix> svd = new ComplexSVD().decompose(src);
        int rank = svd.getRank();

        return src.numRows-rank==0 ?
                new CMatrix(src.numCols, 1) :
                svd.getU().getSlice(0, src.numRows, rank, src.numRows);
    }


    /**
     * Checks if two sets of vectors, stored as the columns of matrices, span the same space. That is,
     * if each column of both matrices can be expressed as a linear combination of the columns of the other matrix.
     * @param src1 Matrix containing as its columns the first set of vectors.
     * @param src2 Matrix containing as its columns the second set of vectors.
     * @return True if the column vectors of {@code src1} and {@code src2} span the same space.
     */
    public static boolean hasEqualSpan(Matrix src1, Matrix src2) {
        boolean result;

        RealLstsqSolver lstsq = new RealLstsqSolver();
        double tol = Flag4jConstants.EPS_F64; // Tolerance for considering a norm zero.
        Vector solution, col;
        result = true;

        // Check that each column of src2 is a linear combination of the columns in src1.
        for(int j=0; j<src2.numCols; j++) {
            col = src2.getCol(j);
            solution = lstsq.solve(src1, col);

            double norm = VectorNorms.norm(src1.mult(solution).sub(col).data);
            if(norm > tol) {
                // Then the least squares solution does not provide an "exact" solution.
                // Hence, the column of src2 cannot be expressed as a linear combination of the columns of src1
                result = false;
                break;
            }
        }

        return result;
    }


    /**
     * Checks if two sets of vectors, stored as the columns of matrices, span the same space. That is,
     * if each column of both matrices can be expressed as a linear combination of the columns of the other matrix.
     * @param src1 Matrix containing as its columns the first set of vectors.
     * @param src2 Matrix containing as its columns the second set of vectors.
     * @return True if the column vectors of {@code src1} and {@code src2} span the same space.
     */
    public static boolean hasEqualSpan(CMatrix src1, CMatrix src2) {
        boolean result;

        ComplexLstsqSolver lstsq = new ComplexLstsqSolver();
        double tol = Flag4jConstants.EPS_F64; // Tolerance for considering a norm zero.
        CVector solution, col;
        result = true;

        // Check that each column of src2 is a linear combination of the columns in src1.
        for(int j=0; j<src2.numCols; j++) {
            col = src2.getCol(j);
            solution = lstsq.solve(src1, col);

            double norm = src1.mult(solution).sub(col).norm();
            if(norm > tol) {
                // Then the least squares solution does not provide an "exact" solution.
                // Hence, the column of src2 cannot be expressed as a linear combination of the columns of src1
                result = false;
                break;
            }
        }

        return result;
    }
}
