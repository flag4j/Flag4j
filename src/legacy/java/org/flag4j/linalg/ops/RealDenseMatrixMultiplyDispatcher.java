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

package org.flag4j.linalg.ops;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.ops.dense.real.RealDenseMatMult;
import org.flag4j.linalg.ops.dense.real.RealDenseMatMultTranspose;
import org.flag4j.util.ValidateParameters;

import java.util.HashMap;
import java.util.Map;

// TODO: The selection algorithm should be redesigned. All threshold values should be specified in a config file.
// TODO: Investigate the performance of utilizing a selection cache which caches the implementation to be used for recent matrix sizes.
//  Should probably be implemented as a LRU cache (or similar) by extending LinkedHashMap

/**
 * Singleton class which stores a map of all viable real dense matrix multiply algorithms and uses that map to dispatch
 * a real dense matrix multiply problem to the appropriate algorithm.
 */
public final class RealDenseMatrixMultiplyDispatcher {

    /**
     * Singleton instance of this class.
     */
    private static RealDenseMatrixMultiplyDispatcher singletonInstance = null;
    /**
     * lookup table for all real dense matrix multiply algorithms.
     */
    private final Map<AlgorithmNames, RealDenseTensorBinaryOperation> algorithmMap;

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
     * Creates an instance containing a map of all viable real dense matrix multiply algorithms.
     */
    private RealDenseMatrixMultiplyDispatcher() {
        AlgorithmNames[] names = AlgorithmNames.values();
        algorithmMap = new HashMap<>();

        RealDenseTensorBinaryOperation[] algorithms = {
                RealDenseMatMult::standard,
                RealDenseMatMult::reordered,
                RealDenseMatMult::blocked,
                RealDenseMatMult::blockedReordered,
                RealDenseMatMult::concurrentStandard,
                RealDenseMatMult::concurrentReordered,
                RealDenseMatMult::concurrentBlocked,
                RealDenseMatMult::concurrentBlockedReordered,
                RealDenseMatMult::standardVector,
                RealDenseMatMult::blockedVector,
                RealDenseMatMult::concurrentStandardVector,
                RealDenseMatMult::concurrentBlockedVector,
                RealDenseMatMultTranspose::multTranspose,
                RealDenseMatMultTranspose::multTransposeBlocked,
                RealDenseMatMultTranspose::multTransposeConcurrent,
                RealDenseMatMultTranspose::multTransposeBlockedConcurrent,
        };

        for(int i = 0; i< algorithms.length; i++)
            algorithmMap.put(names[i], algorithms[i]);
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static synchronized RealDenseMatrixMultiplyDispatcher getInstance()  {
        singletonInstance = (singletonInstance == null) ? new RealDenseMatrixMultiplyDispatcher() : singletonInstance;
        return singletonInstance;
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @param A First matrix in the multiplication.
     * @param B Second matrix in the multiplication.
     * @return The result of the matrix multiplication.
     */
    public static double[] dispatch(Matrix A, Matrix B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape); // Ensure matrix shapes are conducive to matrix multiplication.

        RealDenseMatrixMultiplyDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(A.shape, B.shape);
        return dispatcher.algorithmMap.get(name).apply(A.data, A.shape, B.data, B.shape);
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of the matrix multiplication.
     */
    public static double[] dispatch(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        ValidateParameters.ensureMatMultShapes(shape1, shape2); // Ensure matrix shapes are conducive to matrix multiplication.

        RealDenseMatrixMultiplyDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(shape1, shape2);
        return dispatcher.algorithmMap.get(name).apply(src1, shape1, src2, shape2);
    }


    /**
     * Dispatches a matrix multiply-transpose problem equivalent to A.mult(B.T()) to the appropriate algorithm based
     * on the size of the matrices.
     * @param A First matrix in the multiplication.
     * @param B Second matrix in the multiplication and the matrix to transpose.
     * @return The matrix multiply-transpose result of {@code A} and {@code B}.
     */
    public static double[] dispatchTranspose(Matrix A, Matrix B) {
        ValidateParameters.ensureArrayLengthsEq(A.numCols, B.numCols);

        RealDenseMatrixMultiplyDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithmTranspose(A.shape);
        return dispatcher.algorithmMap.get(name).apply(A.data, A.shape, B.data, B.shape);
    }



    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    static AlgorithmNames selectAlgorithm(Shape shape1, Shape shape2) {
        AlgorithmNames name;

        int rows1 = shape1.getSize(0);
        int cols1 = shape1.getSize(1);

        // Determine the matrix shape.
        int matrixShape;
        if (getRatio(shape1) >= SQUARENESS_RATIO) {
            matrixShape = 0; // Approximately Square
        } else if (rows1 > cols1) {
            matrixShape = 1; // More rows than columns
        } else {
            matrixShape = 2; // More columns than rows
        }

        // Determine the algorithm name based on the matrix shape and size.
        switch (matrixShape) {
            case 0: // Square
                if(rows1 < SEQUENTIAL_SWAPPED_THRESHOLD) {
                    name = AlgorithmNames.REORDERED;
                } else if (rows1 < CONCURRENT_SWAPPED_THRESHOLD) {
                    name = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    name = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
                break;

            case 1: // More rows than columns
                if(rows1 <= 100 && cols1 <= 5) {
                    name = AlgorithmNames.REORDERED;
                } else {
                    name = AlgorithmNames.CONCURRENT_REORDERED;
                }
                break;

            case 2: // More columns than rows
                if(cols1 <= 100) {
                    name = (rows1 <= 20) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else if(cols1 <= 500) {
                    name = (rows1 <= 10) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    if(rows1 <= 5) {
                        name = AlgorithmNames.REORDERED;
                    } else if(rows1 <= 50) {
                        name = AlgorithmNames.CONCURRENT_STANDARD;
                    } else {
                        name = AlgorithmNames.CONCURRENT_REORDERED;
                    }
                }
                break;
            default:
                name = AlgorithmNames.BLOCKED_REORDERED; // Default to blocked reordered algorithm.
        }

        return name;
    }


    /**
     * Selects the matrix multiplication-transpose algorithm to use based on the size of the first matrix.
     * @param shape Shape of the first matrix.
     * @return The algorithm to use to compute the matrix multiplication-transpose.
     */
    static AlgorithmNames selectAlgorithmTranspose(Shape shape) {
        AlgorithmNames name;
        int rows = shape.getSize(0);

        // TODO: This currently only works well if both matrices are square.

        if(rows < 40) {
            name = AlgorithmNames.MULT_T;
        } else if(rows < 55) {
            name = AlgorithmNames.MULT_T_BLOCKED;
        } else if(rows < 1200) {
            name = AlgorithmNames.MULT_T_CONCURRENT;
        } else {
            name = AlgorithmNames.MULT_T_BLOCKED_CONCURRENT;
        }

        return name;
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

        double ratio = Math.abs(numRows-numCols);
        return 1-ratio/Math.max(numRows, numCols);
    }


    /**
     * Simple enum class containing all possible choices of real dense matrix multiply algorithms.
     */
    protected enum AlgorithmNames {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR, MULT_T, MULT_T_BLOCKED,
        MULT_T_CONCURRENT, MULT_T_BLOCKED_CONCURRENT
    }
}
