/*
 * MIT License
 *
 * Copyright (c) 2026. Jacob Watters
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

import java.util.Objects;

// TODO APIv3: Add Javadocs.
public class PolynomialF64 implements RingElement<PolynomialF64> {

    private final double[] coefficients;

    /// Constructs a polynomial with real `double` coefficients.
    /// @param coefficients The coefficients in ascending order. The first coefficient is the constant term, and the last coefficient
    /// is the `n`th term where `n = coefficients.length`. Zero coefficients *must* be included.
    public PolynomialF64(double... coefficients) {
        Objects.requireNonNull(coefficients, "coefficients cannot be null");
        this.coefficients = coefficients;

        Math.fma(coefficients[0], coefficients[1], coefficients[2]);
    }


    /**
     * Computes difference of two elements of this ring.
     *
     * @param b Second ring element in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public PolynomialF64 sub(PolynomialF64 b) {
        // TODO: Implement this method
        return null;
    }


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is the additive inverse for a ring element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this ring element.
     */
    @Override
    public PolynomialF64 negate() {
        // TODO: Implement this method
        return null;
    }


    /**
     * Sums two elements of this semiring (associative and commutative).
     *
     * @param b Second semiring element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public PolynomialF64 add(PolynomialF64 b) {
        // TODO: Implement this method
        return null;
    }


    /**
     * Multiplies two elements of this semiring (associative).
     *
     * @param b Second semiring element in product.
     *
     * @return The product of this semiring element and {@code b}.
     */
    @Override
    public PolynomialF64 mult(PolynomialF64 b) {
        // TODO: Implement this method
        return null;
    }


    /**
     * <p>Checks if this value is an additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return True if this value is an additive identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        // TODO: Implement this method
        return false;
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return True if this value is a multiplicative identity for this semiring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        // TODO: Implement this method
        return false;
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public PolynomialF64 getZero() {
        // TODO: Implement this method
        return null;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public PolynomialF64 getOne() {
        // TODO: Implement this method
        return null;
    }


    /**
     * Compares this element of the semiring with {@code b}.
     *
     * @param b Second element of the semiring.
     *
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
    public int compareTo(PolynomialF64 b) {
        // TODO: Implement this method
        return 0;
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        // TODO: Implement this method
        return 0;
    }
}
