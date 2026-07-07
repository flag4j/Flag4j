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

/**
 * <p>Provides implementations for dense tensors, matrices, and vectors.
 * 
 * <p>This package includes:
 * <ul>
 *   <li><b>Vectors:</b> One-dimensional ndarrays representing mathematical vectors.</li>
 *   <li><b>Matrices:</b> Two-dimensional ndarrays representing mathematical matrices.</li>
 *   <li><b>Tensors:</b> Multi-dimensional ndarrays representing higher-order tensors.</li>
 * </ul>
 *
 * <p>The implementations support various numeric types and algebraic structures:
 * <ul>
 *   <li><b>Real Numbers:</b> Dense ndarrays of real numbers (double-precision floating-point).</li>
 *   <li><b>Complex Numbers:</b> Dense ndarrays of complex numbers ({@link org.flag4j.numbers.Complex64} or
 *   {@link org.flag4j.numbers.Complex128}).</li>
 *   <li><b>Generic FieldElement Elements:</b> Dense ndarrays parameterized over a generic
 *   {@link org.flag4j.numbers.FieldElement FieldElement}.</li>
 *   <li><b>RingElement and SemiringElement Elements:</b> Dense ndarrays parameterized over {@link org.flag4j.numbers.RingElement RingElement}
 *   and {@link org.flag4j.numbers.SemiringElement SemiringElement} elements.</li>
 * </ul>
 *
 * <p>The package provides optimized implementations for numerical computations, including basic arithmetic operations,
 * linear algebra routines, and tensor operations. These implementations are designed for performance and ease of use.
 *
 * <h2>Package Contents</h2>
 * <ul>
 *   <li>{@link org.flag4j.arrays.dense.Vector Vector} - Dense vector of real numbers (backed by primitive {@code double} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.Matrix Matrix} - Dense matrix of real numbers (backed by primitive {@code double} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.Tensor Tensor} - Dense tensor of real numbers (backed by primitive {@code double} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.CVector CVector} - Dense vector of complex numbers (backed by
 *   {@link org.flag4j.numbers.Complex128 Complex128} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.CMatrix CMatrix} - Dense matrix of complex numbers (backed by
 *   {@link org.flag4j.numbers.Complex128 Complex128} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.CTensor CTensor} - Dense tensor of complex numbers (backed by
 *   {@link org.flag4j.numbers.Complex128 Complex128} array).</li>
 *   <li>{@link org.flag4j.arrays.dense.FieldVector FieldVector&lt;T&gt;} - Dense vector parameterized over a field element
 *   {@code T extends FieldElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.FieldMatrix FieldMatrix&lt;T&gt;} - Dense matrix parameterized over a field element
 *   {@code T extends FieldElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.FieldTensor FieldTensor&lt;T&gt;} - Dense tensor parameterized over a field element
 *   {@code T extends FieldElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.RingVector RingVector&lt;T&gt;} - Dense vector parameterized over a ring element
 *   {@code T extends RingElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.RingMatrix RingMatrix&lt;T&gt;} - Dense matrix parameterized over a ring element
 *   {@code T extends RingElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.RingTensor RingTensor&lt;T&gt;} - Dense tensor parameterized over a ring element
 *   {@code T extends RingElement<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.SemiringVector SemiringVector&lt;T&gt;} - Dense vector parameterized over a semiring element
 *   {@code T extends semiring<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.SemiringMatrix SemiringMatrix&lt;T&gt;} - Dense matrix parameterized over a semiring element
 *   {@code T extends semiring<T>}.</li>
 *   <li>{@link org.flag4j.arrays.dense.SemiringTensor SemiringTensor&lt;T&gt;} - Dense tensor parameterized over a semiring element
 *   {@code T extends semiring<T>}.</li>
 * </ul>
 *
 * @see org.flag4j.arrays.sparse
 */
package org.flag4j.arrays.dense;