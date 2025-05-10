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

package org.flag4j.linalg.ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.dense.Vector;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseMatMult;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.ValidateParameters;

/**
 * Dispatches matrix multiplication to the appropriate algorithm based on the size of the matrices to be multiplied.
 */
public final class MatrixMultiplyDispatcher {

    private MatrixMultiplyDispatcher() {
        // Hide constructor for utility class. of utility class
    }


    /*
        TODO: Move all dispatch methods to their own singleton classes like RealDenseMatrixMultiplyDispatcher.
     */


    /**
     * Ration measuring squareness. the closer to one, the more square the matrix is.
     */
    private static final double SQUARENESS_RATIO = 0.75;
    /**
     * Threshold for small matrices which should be multiplied using the standard ikj algorithm.
     */
    private static final int SEQUENTIAL_SWAPPED_THRESHOLD = 40;
    /**
     * Threshold for matrices to use the concurrent ikj algorithm.
     */
    private static final int CONCURRENT_SWAPPED_THRESHOLD = 3072;


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static Complex128[] dispatch(Matrix A, CVector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        ValidateParameters.ensureMatMultShapes(A.shape, bMatShape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows];

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static CVector dispatch(CMatrix A, Vector b) {
        ValidateParameters.ensureMatMultShapes(A.shape, b.shape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows];

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, b.data, b.shape, dest);
                break;
        }

        return new CVector(dest);
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static Complex128[] dispatch(Matrix A, CMatrix B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows*B.numCols];

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                RealFieldDenseMatMult.standard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case REORDERED:
                RealFieldDenseMatMult.reordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED:
                RealFieldDenseMatMult.blocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_REORDERED:
                RealFieldDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD:
                RealFieldDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_REORDERED:
                RealFieldDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED:
                RealFieldDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                RealFieldDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static Complex128[] dispatch(CMatrix A, Matrix B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows*B.numCols];

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                RealFieldDenseMatMult.standard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case REORDERED:
                RealFieldDenseMatMult.reordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED:
                RealFieldDenseMatMult.blocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_REORDERED:
                RealFieldDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD:
                RealFieldDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_REORDERED:
                RealFieldDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED:
                RealFieldDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                RealFieldDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmName chooseAlgorithmRealComplex(Shape shape1, Shape shape2) {
        AlgorithmName algorithm;

        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);

        // TODO: Extract constants to final variables.
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(rows1<=40) {
                algorithm = AlgorithmName.REORDERED;
            } else if(rows1<=225) {
                algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
            /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
            better cache performance on modern systems */
                algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100) {
                if(cols1<=2) algorithm = AlgorithmName.REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
                if(cols1<=45) algorithm = AlgorithmName.CONCURRENT_REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=15) {
                    algorithm = AlgorithmName.REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=15) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=100) {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
                }
            } else {
                if(rows1<=2) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=15){
                    algorithm = AlgorithmName.BLOCKED_REORDERED;
                } else if(rows1<=150) {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
                }
            }
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmName chooseAlgorithmRealComplexVector(Shape shape) {
        int rows = shape.getSize(0);

        if(rows<=600) return AlgorithmName.STANDARD_VECTOR;
        else return AlgorithmName.CONCURRENT_BLOCKED_VECTOR;
    }


    /**
     * Computes the squareness ratio of a matrix. This is a value between 0 and 1 with 1 being perfectly
     * square and 0 being a row/column vector.
     * @param shape Shape of the matrix to compute the squareness ratio of.
     * @return The squareness ratio for the specified shape.
     */
    private static double getRatio(Shape shape) {
        int numRows = shape.getSize(0);
        int numCols = shape.getSize(1);

        double ratio = Math.abs(numRows-numCols) / Math.max(numRows, numCols);
        return 1 - ratio;
    }


    /**
     * Simple enum class containing all possible choices of matrix multiply algorithms.
     */
    private enum AlgorithmName {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR, MULT_T, MULT_T_BLOCKED,
        MULT_T_CONCURRENT, MULT_T_BLOCKED_CONCURRENT
    }
}
