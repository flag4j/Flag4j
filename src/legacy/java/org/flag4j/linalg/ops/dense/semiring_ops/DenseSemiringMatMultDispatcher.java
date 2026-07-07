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

package org.flag4j.linalg.ops.dense.semiring_ops;

// TODO: The selection algorithm should be redesigned. All threshold values should be specified in a config file.
// TODO: Investigate the performance of utilizing a selection cache which caches the implementation to be used for recent matrix sizes.
//  Should probably be implemented as a LRU cache (or similar) by extending LinkedHashMap

import org.flag4j.arrays.Shape;
import org.flag4j.numbers.SemiringElement;
import org.flag4j.util.ValidateParameters;

import java.util.HashMap;
import java.util.Map;

/**
 * Singleton class which stores a map of all viable dense {@link SemiringElement}
 * matrix multiply algorithms and uses that map to dispatch a dense {@link SemiringElement} matrix
 * multiply problem to the appropriate algorithm.
 */
public final class DenseSemiringMatMultDispatcher {

    // TODO: All thresholds in this class are currently HIGHLY speculative. The "best" default thresholds may vary significantly
    //  depending on the particular concrete implementation of the SemiringElement interface. Detailed benchmarking should be done to
    //  determine good baseline thresholds which should also be configurable in a file.

    /**
     * Singleton instance of this class.
     */
    private static DenseSemiringMatMultDispatcher singletonInstance = null;
    /**
     * lookup table for all real dense matrix multiply algorithms.
     */
    private final Map<AlgorithmNames, DenseSemiringTensorBinaryOperation> algorithmMap;

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
    private DenseSemiringMatMultDispatcher() {
        AlgorithmNames[] names = AlgorithmNames.values();
        algorithmMap = new HashMap<>();

        DenseSemiringTensorBinaryOperation[] algorithms = {
                DenseSemiringMatMult::standard,
                DenseSemiringMatMult::reordered,
                DenseSemiringMatMult::blocked,
                DenseSemiringMatMult::blockedReordered,
                DenseSemiringMatMult::concurrentStandard,
                DenseSemiringMatMult::concurrentReordered,
                DenseSemiringMatMult::concurrentBlocked,
                DenseSemiringMatMult::concurrentBlockedReordered,
                DenseSemiringMatMult::standardVector,
                DenseSemiringMatMult::blockedVector,
                DenseSemiringMatMult::concurrentStandardVector,
                DenseSemiringMatMult::concurrentBlockedVector,
                DenseSemiringMatMultTranspose::multTranspose,
                DenseSemiringMatMultTranspose::multTransposeBlocked,
                DenseSemiringMatMultTranspose::multTransposeConcurrent,
                DenseSemiringMatMultTranspose::multTransposeBlockedConcurrent,
        };

        for(int i=0; i< algorithms.length; i++)
            algorithmMap.put(names[i], algorithms[i]);
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static synchronized DenseSemiringMatMultDispatcher getInstance()  {
        singletonInstance = (singletonInstance == null) ? new DenseSemiringMatMultDispatcher() : singletonInstance;
        return singletonInstance;
    }


    /**
     * Dispatches a matrix-vector multiplication problem to the appropriate algorithm based on the size of the matrix and vector.
     * @param src1 Entries of the matrix in the matrix-vector multiplication problem.
     * @param shape1 Shape of the matrix {@code src1}.
     * @param src2 Entries of the vector in the matrix-vector multiplication problem.
     * @param shape2 Shape of the vector {@code src2}.
     * @param dest Array to store the result of the matrix-vector multiplication.
     */
    public static <T extends SemiringElement<T>> void dispatchVector(T[] src1, Shape shape1,
                                                                     T[] src2, Shape shape2,
                                                                     T[] dest) {
        Shape bMatShape = new Shape(shape2.getSize(0), 1); // Shape of column vector.
        ValidateParameters.ensureMatMultShapes(shape1, bMatShape);

        AlgorithmNames algorithm = selectAlgorithmVector(shape1);

        switch(algorithm) {
            case STANDARD_VECTOR:
                DenseSemiringMatMult.standardVector(src1, shape1, src2, bMatShape, dest);
                break;
            case BLOCKED_VECTOR:
                DenseSemiringMatMult.blockedVector(src1, shape1, src2, bMatShape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                DenseSemiringMatMult.concurrentStandardVector(src1, shape1, src2, bMatShape, dest);
                break;
            default:
                DenseSemiringMatMult.concurrentBlockedVector(src1, shape1, src2, bMatShape, dest);
                break;
        }
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiplication in.
     */
    public static <T extends SemiringElement<T>> void dispatch(T[] src1, Shape shape1,
                                                               T[] src2, Shape shape2,
                                                               T[] dest) {
        ValidateParameters.ensureMatMultShapes(shape1, shape2); // Ensure matrix shapes are conducive to matrix multiplication.

        DenseSemiringMatMultDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(shape1, shape2);
        dispatcher.algorithmMap.get(name).apply(src1, shape1, src2, shape2, dest);
    }


    /**
     * Dispatches a matrix multiply-transpose problem equivalent to {@code src1.mult(src2.T())} to the appropriate algorithm based
     * on the size of the matrices.
     * @param src1 Entries of the first matrix in the multiplication.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix in the multiplication and the matrix to transpose.
     * @param shape2 Shape of the second matrix.
     * @param dest Array to store the result of the matrix multiply-transpose in.
     */
    public static <T extends SemiringElement<T>> void dispatchTranspose(T[] src1, Shape shape1,
                                                                        T[] src2, Shape shape2,
                                                                        T[] dest) {
        ValidateParameters.ensureArrayLengthsEq(shape1.getSize(1), shape2.getSize(1));

        DenseSemiringMatMultDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithmTranspose(shape1);
        dispatcher.algorithmMap.get(name).apply(src1, shape1, src2, shape2, dest);
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

        // TODO: This only verified to work well if both matrices are square.
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
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames selectAlgorithmVector(Shape shape) {
        if(shape.getSize(0) <=600) return AlgorithmNames.STANDARD_VECTOR;
        else return AlgorithmNames.CONCURRENT_BLOCKED_VECTOR;
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


    /**
     * Specifies a binary operation on two dense {@link SemiringElement} tensors.
     */
    interface DenseSemiringTensorBinaryOperation {

        /**
         * Applies the specified binary operation on the two dense tensors.
         * @param src1 Entries of the first tensor.
         * @param shape1 Shape of the first tensor.
         * @param src2 Entries of the second tensor.
         * @param shape2 Shape of the second tensor.
         * @param dest Array to store the result of the binary operation of the two tensors in.
         */
        public <T extends SemiringElement<T>> void apply(T[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest);
    }
}
