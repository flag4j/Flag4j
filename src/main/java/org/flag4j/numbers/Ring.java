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

package org.flag4j.numbers;

/**
 * Defines a mathematical ring structure and specifies the operations that ring elements must support.
 *
 * <p>A <b>ring</b> is an algebraic structure consisting of a set <b>R</b> equipped with two binary operations:
 * addition (+) and multiplication (*). Rings generalize {@link Field fields} by not requiring every non-zero-element to have a
 * multiplicative inverse, and multiplication may not be commutative.
 *
 * <h2>Formal Definition:</h2>
 * <p>For all elements {@code a}, {@code b}, and {@code c} in <b>R</b>, the following properties must hold:
 * <ul>
 *   <li><b>Addition is associative:</b> {@code a + (b + c) = (a + b) + c}</li>
 *   <li><b>Addition is commutative:</b> {@code a + b = b + a}</li>
 *   <li><b>Additive identity exists:</b> There exists an element {@code 0} in <b>R</b> such that {@code a + 0 = a}</li>
 *   <li><b>Existence of additive inverses:</b> For every {@code a} in <b>R</b>, there exists an element {@code -a}
 *   in <b>R</b> such that {@code a + (-a) = 0}</li>
 *   <li><b>Multiplication is associative:</b> {@code a * (b * c) = (a * b) * c}</li>
 *   <li><b>Distributivity of multiplication over addition:</b>
 *     <ul>
 *       <li><b>Left distributivity:</b> {@code a * (b + c) = (a * b) + (a * c)}</li>
 *       <li><b>Right distributivity:</b> {@code (a + b) * c = (a * c) + (b * c)}</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * <h2>Implementations:</h2>
 * <p>Implementations of the {@code Ring} interface should ensure that instances are immutable. This means
 * that all operations should return new instances rather than modifying existing ones. Immutability guarantees
 * thread safety and consistent behavior across different contexts.
 *
 * <p>The {@link #compareTo(Semiring)} method should implement some ordering (total or partial) on the semiring.
 *
 * <p>Further, implementations should ensure that the equality and hash code methods are consistent with the semiring's equality
 * definition.
 *
 * <h2>Interface Methods:</h2>
 * <p>The {@code Ring} interface specifies the following methods that ring elements must implement:
 * <ul>
 *   <li>{@link #add(Semiring)}: Performs the addition operation, returning a new ring element.</li>
 *   <li>{@link #mult(Semiring)}: Performs the multiplication operation, returning a new ring element.</li>
 *   <li>{@link #sub(Ring)}: Performs the subtraction operation, defined as addition with the additive inverse.</li>
 *   <li>{@link #addInv()}: Returns the additive inverse of this element.</li>
 *   <li>{@link #isZero()}: Checks if the element is the additive identity (zero-element).</li>
 *   <li>{@link #isOne()}: Checks if the element is the multiplicative identity (one element), if it exists.</li>
 *   <li>{@link #getZero()}: Returns the additive identity element of the ring.</li>
 *   <li>{@link #getOne()}: Returns the multiplicative identity element of the ring, if it exists.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Assume T is a concrete implementation of Ring<T>
 * T a = ...; // Initialize element a
 * T b = ...; // Initialize element b
 *
 * T sum = a.add(b);        // Perform addition in the ring
 * T difference = a.sub(b); // Perform subtraction in the ring
 * T product = a.mult(b);   // Perform multiplication in the ring
 * T negation = a.addInv(); // Get the additive inverse of a
 *
 * boolean isZero = a.isZero(); // Check if a is the additive identity
 * boolean isOne = b.isOne();   // Check if b is the multiplicative identity
 * }</pre>
 *
 * <h2>Examples of Rings:</h2>
 * <ul>
 *   <li><b>Integers (ℤ):</b> The set of integers with usual addition and multiplication forms a ring.</li>
 *   <li><b>Polynomials:</b> Polynomials with coefficients in a field form a ring under polynomial addition and multiplication.</li>
 *   <li><b>Matrix Rings:</b> Square matrices of a given size over a ring form a ring under matrix addition and multiplication.</li>
 * </ul>
 *
 * @see Semiring
 * @see Field
 *
 * @param <T> the type of the ring element.
 */
public interface Ring<T extends Ring<T>> extends Semiring<T> {

    /**
     * Computes difference of two elements of this ring.
     * @param b Second ring element in difference.
     * @return The difference of this ring element and {@code b}.
     */
    T sub(T b);


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this ring element.
     */
    T addInv();


    /**
     * <p>Computes the absolute value of this ring element.
     *
     * @return The absolute value of this ring element.
     * @implNote By default, this is implemented as {@code return }{@link #mag()}{@code ;}
     */
    default double abs() {
        return mag();
    }


    /**
     * Computes the magnitude of this ring element.
     * @return The magnitude of this ring element.
     */
    default double mag() {
        throw new UnsupportedOperationException("Magnitude/absolute value is not defined for this algebraic object: "
                + getClass().getName() + ".");
    }


    /**
     * Computes the conjugation of this ring's element.
     * @return The conjugation of this ring's element.
     * @implNote The default implementation of this method simply returns this rings element.
     */
    default T conj() {
        throw new UnsupportedOperationException("Conjugation is not defined for this algebraic object: "
                + getClass().getName() + ".");
    }
}
