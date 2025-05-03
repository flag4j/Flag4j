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

/**
 * <p>Contains implementations for sparse tensors, matrices, and vectors.
 *
 * <p>A sparse tensor, matrix, or vector is an array where most elements are zero. There is no strict definition for the
 * proportion of zero-values elements within the array for it to be considered sparse but generally there should be <em>many</em>
 * times more zeros than non-zero values.
 *
 * <p>Sparse arrays can offer advantages such as reduced memory usage and, in many cases, improved computational performance.
 * These benefits depend on the degree of sparsity and the algorithms used.
 *
 * <p>It should be noted that certain ops on sparse arrays can result in catastrophic loss of sparsity significantly increasing
 * the number of non-zero-elements, resulting in higher memory consumption and degraded performance. These effects can be
 * particularly problematic for algorithms optimized for sparse data.
 * Such ops will be documented and should be used with care.
 *
 * <p>This package includes implementations for arrays with real and complex values, as well as generalized support for
 * {@link org.flag4j.numbers.Semiring semirings},
 * {@link org.flag4j.numbers.Ring rings}, and {@link org.flag4j.numbers.Field fields}.
 * Additionally, some specialized sparse matrix types are provided with limited support for ops:
 * {@link org.flag4j.arrays.sparse.PermutationMatrix permutation matrix} and the {@link org.flag4j.arrays.sparse.SymmTriDiag
 * symmetric tri-diagonal matrix}.
 *
 * @see org.flag4j.arrays.dense
 */
package org.flag4j.arrays.sparse;