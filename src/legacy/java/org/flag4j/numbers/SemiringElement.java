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

package org.flag4j.numbers;


import java.io.Serializable;

/**
 * Defines a mathematical semiring structure and specifies the operations that semiring elements must support.
 *
 * <p>A <b>semiring</b> is an algebraic structure consisting of a set <b>R</b> equipped with two binary operations:
 * addition (+) and multiplication (*). The semiring generalizes the concept of a {@link RingElement ring} but does not require the
 * existence of additive inverses (i.e., negative elements). As a result, subtraction is not generally defined within a semiring.
 *
 * <h2>Formal Definition:</h2>
 * <p>For all elements {@code a}, {@code b}, and {@code c} in <b>R</b>, the following properties must hold:
 * <ul>
 *   <li><b>Addition is associative:</b> {@code a + (b + c) = (a + b) + c}</li>
 *   <li><b>Addition is commutative:</b> {@code a + b = b + a}</li>
 *   <li><b>Additive identity exists:</b> There exists an element {@code 0} in <b>R</b> such that {@code a + 0 = a}</li>
 *   <li><b>Multiplication is associative:</b> {@code a * (b * c) = (a * b) * c}</li>
 *   <li><b>Multiplicative identity exists:</b> There exists an element {@code 1} in <b>R</b>
 *   such that {@code a * 1 = a}</li>
 *   <li><b>Multiplication distributes over addition:</b>
 *     <ul>
 *       <li><b>Left distributivity:</b> {@code a * (b + c) = (a * b) + (a * c)}</li>
 *       <li><b>Right distributivity:</b> {@code (a + b) * c = (a * c) + (b * c)}</li>
 *     </ul>
 *   </li>
 *   <li><b>Zero is absorbing for multiplication:</b> {@code a * 0 = 0 * a = 0}</li>
 * </ul>
 *
 * <h2>Implementations:</h2>
 * <p>Implementations of the {@code SemiringElement} interface should ensure that instances are immutable. This means
 * that all operations should return new instances rather than modifying existing ones. Immutability guarantees
 * thread safety and consistent behavior across different contexts.
 *
 * <p> The {@link #compareTo(SemiringElement)} method should implement some ordering (total or partial) on the semiring.
 *
 * <p>Further, implementations should ensure that the equality and hash code methods are consistent with the semiring's equality
 * definition.
 *
 * <h2>Examples of Semirings:</h2>
 * <ul>
 *   <li><b>Natural Numbers (ℕ):</b> The set of natural numbers with usual addition and multiplication forms a semiring.</li>
 *   <li><b>Boolean SemiringElement:</b> The set {{@code false}, {@code true}} with logical OR as addition and logical AND as multiplication.</li>
 *   <li><b>Tropical SemiringElement:</b> The set of real numbers extended with infinity, where addition is defined as taking the minimum
 *   (or maximum) and multiplication as the standard real number addition.</li>
 * </ul>
 *
 * <h2>Interface Methods:</h2>
 * <p>The {@code SemiringElement} interface specifies the following methods that semiring elements must implement:
 * <ul>
 *   <li>{@link #add(SemiringElement)}: Performs the addition operation, returning a new semiring element.</li>
 *   <li>{@link #mult(SemiringElement)}: Performs the multiplication operation, returning a new semiring element.</li>
 *   <li>{@link #isZero()}: Checks if the element is the additive identity (zero-element).</li>
 *   <li>{@link #isOne()}: Checks if the element is the multiplicative identity (one element).</li>
 *   <li>{@link #getZero()}: Returns the additive identity element of the semiring.</li>
 *   <li>{@link #getOne()}: Returns the multiplicative identity element of the semiring.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Assume T is a concrete implementation of SemiringElement<T>
 * T a = ...; // Initialize element a
 * T b = ...; // Initialize element b
 *
 * T sum = a.add(b);       // Perform addition in the semiring
 * T product = a.mult(b);  // Perform multiplication in the semiring
 *
 * boolean isZero = a.isZero(); // Check if a is the additive identity
 * boolean isOne = b.isOne();   // Check if b is the multiplicative identity
 * }</pre>
 *
 * @see RingElement
 * @see FieldElement
 *
 * @param <T> the type of the semiring element.
 */
public interface SemiringElement<T extends SemiringElement<T>> extends Comparable<T>, Serializable {

    /**
     * Sums two elements of this semiring (associative and commutative).
     * @param b Second semiring element in sum.
     * @return The sum of this element and {@code b}.
     */
    T add(T b);


    /**
     * Multiplies two elements of this semiring (associative).
     * @param b Second semiring element in product.
     * @return The product of this semiring element and {@code b}.
     */
    T mult(T b);


    /**
     * <p>Checks if this value is an additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return True if this value is an additive identity for this semiring. Otherwise, false.
     */
    boolean isZero();


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return True if this value is a multiplicative identity for this semiring. Otherwise, false.
     */
    boolean isOne();


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    T getZero();


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    T getOne();


    /**
     * Compares this element of the semiring with {@code b}.
     * @param b Second element of the semiring.
     * @return An int value:
     * <ul>
     *     <li>0 if this semiring element is equal to {@code b}.</li>
     *     <li>< 0 if this semiring element is less than {@code b}.</li>
     *     <li>> 0 if this semiring element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two semiring elements are equal, a negative value if and only the semiring
     *     element it was called on is less than {@code b} and positive if and only if the semiring element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    int compareTo(T b);


    /**
     * Converts this semiring value to an equivalent double value.
     * @return A double value equivalent to this semiring element.
     */
    double doubleValue();
}
