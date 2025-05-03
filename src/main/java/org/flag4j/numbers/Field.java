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
 * Defines a mathematical field structure and specifies the operations that field elements must support.
 *
 * <p>A <b>field</b> is an algebraic structure consisting of a set <b>F</b> equipped with two binary operations:
 * addition (+) and multiplication (*). Fields generalize the familiar arithmetic of rational numbers, real numbers,
 * and complex numbers. In a field, both addition and multiplication are commutative, and every non-zero-element has
 * a multiplicative inverse, allowing for division operations.
 *
 * <h2>Formal Definition:</h2>
 * <p>For all elements {@code }, {@code b}, and {@code c} in <b>F</b>, the following properties must hold:
 * <ul>
 *   <li><b>Addition is associative:</b> {@code a + (b + c) = (a + b) + c}</li>
 *   <li><b>Addition is commutative:</b> {@code a + b = b + a}</li>
 *   <li><b>Additive identity exists:</b> There exists an element {@code 0} in <b>F</b> such that {@code a + 0 = a}</li>
 *   <li><b>Existence of additive inverses:</b> For every {@code a} in <b>F</b>, there exists an element {@code -a}
 *   in <b>F</b> such that {@code a + (-a) = 0}</li>
 *   <li><b>Multiplication is associative:</b> {@code a * (b * c) = (a * b) * c}</li>
 *   <li><b>Multiplication is commutative:</b> {@code a * b = b * a}</li>
 *   <li><b>Multiplicative identity exists:</b> There exists an element {@code 1 ≠ 0} in <b>F</b> such that
 *   {@code a * 1 = a}</li>
 *   <li><b>Existence of multiplicative inverses:</b> For every {@code a ≠ 0} in <b>F</b>, there exists an element
 *   <code>a<sup>-1</sup></code> in <b>F</b> such that <code>a * a<sup>-1</sup> = 1</code></li>
 *   <li><b>Distributivity of multiplication over addition:</b>
 *     <ul>
 *       <li><b>Left distributivity:</b> {@code a * (b + c) = (a * b) + (a * c)}</li>
 *       <li><b>Right distributivity:</b> {@code (a + b) * c = (a * c) + (b * c)}</li>
 *     </ul>
 *   </li>
 * </ul>
 *
 * <h6>Extended Operations:</h6>
 *
 * <h2>Implementations:</h2>
 * <p>Implementations of the {@code Field} interface should ensure that instances are immutable. This means
 * that all operations should return new instances rather than modifying existing ones. Immutability guarantees
 * thread safety and consistent behavior across different contexts.
 *
 * <p>While the interface defines the core operations of a mathematical field, it also specifies additional methods
 * that are common and useful in numerical computations, such as arithmetic with real numbers ({@code double} values), and the
 * computation of square roots. These methods are included to facilitate working with numerical fields like the real
 * numbers or complex numbers. There implementation is optional and default methods are provided which throw a
 * {@link UnsupportedOperationException}.
 *
 * <p> The {@link #compareTo(Semiring)} method should implement some ordering (total or partial) on the semiring.
 *
 * <p>Further, implementations should ensure that the equality and hash code methods are consistent with the semiring's equality
 * definition.
 *
 * <h2>Interface Methods:</h2>
 * <p>The {@code Field} interface extends the {@link Ring} interface and specifies additional methods that field elements must implement:
 * <ul>
 *   <li>{@link #add(Semiring)}: Performs the addition operation, returning a new field element.</li>
 *   <li>{@link #sub(Ring)}: Performs the subtraction operation, defined as addition with the additive inverse.</li>
 *   <li>{@link #mult(Semiring)}: Performs the multiplication operation, returning a new field element.</li>
 *   <li>{@link #div(Field)}: Performs the division operation, defined using the multiplicative inverse.</li>
 *   <li>{@link #addInv()}: Returns the additive inverse of this element.</li>
 *   <li>{@link #multInv()}: Returns the multiplicative inverse of this element.</li>
 *   <li>{@link #isZero()}: Checks if the element is the additive identity (zero-element).</li>
 *   <li>{@link #isOne()}: Checks if the element is the multiplicative identity (one element).</li>
 *   <li>{@link #getZero()}: Returns the additive identity element of the field.</li>
 *   <li>{@link #getOne()}: Returns the multiplicative identity element of the field.</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Assume T is a concrete implementation of Field<T>
 * T a = ...; // Initialize element a
 * T b = ...; // Initialize element b
 *
 * T sum = a.add(b);         // Perform addition in the field
 * T difference = a.sub(b);  // Perform subtraction in the field
 * T product = a.mult(b);    // Perform multiplication in the field
 * T quotient = a.div(b);    // Perform division in the field
 * T negation = a.addInv();  // Get the additive inverse of a
 * T reciprocal = a.multInv(); // Get the multiplicative inverse of a
 *
 * boolean isZero = a.isZero(); // Check if a is the additive identity
 * boolean isOne = b.isOne();   // Check if b is the multiplicative identity
 * }</pre>
 *
 * <h2>Examples of Fields:</h2>
 * <ul>
 *   <li><b>Rational Numbers (ℚ):</b> Fractions of integers where addition, subtraction, multiplication, and division are defined.</li>
 *   <li><b>Real Numbers (ℝ):</b> All continuous numbers along the number line, including irrational numbers.</li>
 *   <li><b>Complex Numbers (ℂ):</b> Numbers of the form {@code a + bi}, where {@code i} is the imaginary unit.</li>
 *   <li><b>Finite Fields (𝔽<sub>p</sub>):</b> Fields with a finite number of elements, such as integers modulo a prime number
 *   {@code p}.</li>
 * </ul>
 *
 * @see Ring
 * @see Semiring
 *
 * @param <T> the type of the field element, which should extend {@code Field<T>} to ensure type consistency.
 */
public interface Field<T extends Field<T>> extends Ring<T> {


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    T div(T b);


    /**
     * Sums an element of this field with a real number (associative and commutative).
     * @param b Real element in sum.
     * @return The sum of this element and {@code b}.
     */
    default T add(double b) {
        throw new UnsupportedOperationException("Addition with primitive doubles is not supported for this field: "
                + getClass() + ".");
    }


    /**
     * Computes difference of an element of this field and a real number.
     * @param b Real value in difference.
     * @return The difference of this ring element and {@code b}.
     */
    default T sub(double b) {
        throw new UnsupportedOperationException("Subtraction with primitive doubles is not supported for this field: "
                + getClass() + ".");
    }


    /**
     * Multiplies an element of this field with a real number (associative and commutative).
     * @param b Real number in product.
     * @return The product of this field element and {@code b}.
     */
    default T mult(double b) {
        throw new UnsupportedOperationException("Multiplication with primitive doubles is not supported for this field: "
                + getClass() + ".");
    }


    /**
     * Computes the quotient of an element of this field and a real number.
     * @param b Real number in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    default T div(double b) {
        throw new UnsupportedOperationException("Division with primitive doubles is not supported for this field: "
                + getClass().getName() + ".");
    }


    /**
     * <p>Computes the multiplicative inverse for an element of this field.
     *
     * <p>An element x<sup>-1</sup> is a multiplicative inverse for a filed element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicative identity.
     *
     * @return The multiplicative inverse for this field element.
     */
    T multInv();


    /**
     * Computes the square root of this field element.
     * @return The square root of this field element.
     */
    default T sqrt() {
        throw new UnsupportedOperationException("Square roots are not supported for this field: "
                + getClass() + ".");
    }


    /**
     * Checks if this field element is finite in magnitude.
     * @return True if this field element is finite in magnitude. False otherwise (i.e., infinite, NaN etc.).
     */
    boolean isFinite();


    /**
     * Checks if this field element is infinite in magnitude.
     * @return True if this field element is infinite in magnitude. False otherwise (i.e., finite, NaN, etc.).
     */
    boolean isInfinite();


    /**
     * Checks if this field element is NaN in magnitude.
     * @return True if this field element is NaN in magnitude. False otherwise (i.e., finite, NaN, etc.).
     */
    boolean isNaN();
}
