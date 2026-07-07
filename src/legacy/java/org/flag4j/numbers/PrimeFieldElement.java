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


/**
 * Represents an immutable element of a prime finite field.
 *
 * <p>This class represents an element of the finite field {@code GF(p)}, where
 * {@code p} is a prime number. Each element is stored in its canonical representative
 * form:
 * <pre>
 * 0 <= value < p
 * </pre>
 *
 * <p>Arithmetic is performed modulo {@code p}. That is, for elements {@code a} and
 * {@code b} in {@code GF(p)}:
 *
 * <ul>
 *   <li><b>Addition:</b> a + b = (a + b) mod p.</li>
 *   <li><b>Subtraction:</b> a - b = (a - b) mod p.</li>
 *   <li><b>Multiplication:</b> a * b = (a * b) mod p.</li>
 *   <li><b>Division:</b> a / b = a * b<sup>-1</sup> mod p, for nonzero b.</li>
 * </ul>
 *
 * <p>The additive identity is {@code 0}, and the multiplicative identity is
 * {@code 1}. Every element has an additive inverse, and every nonzero element has
 * a multiplicative inverse.
 *
 * <p>For example, in {@code GF(7)}:
 *
 * <pre>{@code
 * PrimeFieldElement a = new PrimeFieldElement(7, 5);
 * PrimeFieldElement b = new PrimeFieldElement(7, 6);
 *
 * PrimeFieldElement sum = a.add(b);      // 5 + 6 = 11 mod 7 = 4
 * PrimeFieldElement product = a.mult(b); // 5 * 6 = 30 mod 7 = 2
 * PrimeFieldElement inverse = a.multInv(); // 5^-1 mod 7 = 3
 * }</pre>
 *
 * <p>Although all finite fields are also called Galois fields, this class supports
 * only prime fields of the form {@code GF(p)}. It does not represent extension fields
 * such as {@code GF(p^n)} for {@code n > 1}. For a general finite field, see {@link GaloisFieldElement}.
 *
 * <p>Instances are immutable and thread-safe. Arithmetic between elements with
 * different prime moduli is not permitted.
 *
 * <p>The ordering supplied by {@link #compareTo(PrimeFieldElement)} compares canonical
 * integer representatives. It is not an ordering compatible with field arithmetic.
 *
 * @see FieldElement
 */
public class PrimeFieldElement extends FiniteFieldElement<PrimeFieldElement> {

    /**
     * The prime number of this finite field.
     */
    private final int prime;

    /**
     * The value of this field element.
     */
    private final int value;


    /**
     * Constructs a finite field (Galois field) with the specified prime number.
     * @param prime
     */
    public PrimeFieldElement(int prime, int value) {
        this(prime, value, true);
    }


    /**
     * Constructs a finite field (Galois field) with the specified prime number.
     * @param prime
     * @param value
     * @param validatePrime Flag indicating whether to validate that the prime number is actually prime.
     */
    private PrimeFieldElement(int prime, int value, boolean validatePrime) {
        if (validatePrime && !isPrime(prime)) {
            throw new IllegalArgumentException("p must be prime");
        }

        this.prime = prime;
        this.value = Math.floorMod(value, prime); // Ensure value is canonical value.
    }


    /**
     * Gets the prime number of this finite field.
     * @return The prime number of this finite field.
     */
    public int getPrime() {
        return prime;
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public int getValue() {
        return value;
    }


    /**
     * Computes the quotient of two elements of this field.
     *
     * @param b Second field element in quotient.
     *
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public PrimeFieldElement div(PrimeFieldElement b) {
        ensureSameField(this, b);
        return this.mult(b.invert());
    }


    /**
     * <p>Computes the multiplicative inverse for an element of this field.
     *
     * <p>An element x<sup>-1</sup> is a multiplicative inverse for a filed element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicative identity.
     *
     * @return The multiplicative inverse for this field element.
     * @throws ArithmeticException If this value is zero.
     */
    @Override
    public PrimeFieldElement invert() {
        if (this.value == 0) {
            throw new ArithmeticException("Zero has no multiplicative inverse.");
        }

        return new PrimeFieldElement(prime, modInverse(this.value, this.prime), false);
    }


    /**
     * <p>Checks if this field element is finite in magnitude.
     * <p>Note: This is <em>always</em> {@code true} for finite fields.
     *
     * @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.).
     */
    @Override
    public boolean isFinite() {
        return true;
    }


    /**
     * <p>Checks if this field element is infinite in magnitude.
     * <p>Note: This is <em>always</em> {@code false} for finite fields.
     *
     * @return {@code true} if this field element is infinite in magnitude; {@code false} otherwise (i.e., finite, NaN, etc.).
     */
    @Override
    public boolean isInfinite() {
        return false;
    }


    /**
     * Checks if this field element is NaN in magnitude.
     * <p>Note: This is <em>always</em> {@code false} for finite fields.
     *
     * @return {@code true} if this field element is NaN in magnitude; {@code false} otherwise (i.e., finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return false;
    }


    /**
     * Computes difference of two elements of this ring.
     *
     * @param b Second ring element in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public PrimeFieldElement sub(PrimeFieldElement b) {
        ensureSameField(this, b);
        return this.add(b.negate());
    }


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this ring element.
     */
    @Override
    public PrimeFieldElement negate() {
        return new PrimeFieldElement(prime, value == 0 ? 0 : prime - value);
    }


    /**
     * Sums two elements of this semiring (associative and commutative).
     *
     * @param b Second semiring element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public PrimeFieldElement add(PrimeFieldElement b) {
        ensureSameField(this, b);
        return new PrimeFieldElement(
                prime,
                // Cast to long and mod by prime before casting back to integer to reduce overflows.
                (int) (((long) value + b.value) % prime),
                false
        );
    }


    /**
     * Multiplies two elements of this semiring (associative).
     *
     * @param b Second semiring element in the product.
     *
     * @return The product of this semiring element and {@code b}.
     */
    @Override
    public PrimeFieldElement mult(PrimeFieldElement b) {
        ensureSameField(this, b);
        return new PrimeFieldElement(
                prime,
                // Cast to long and mod by prime before casting back to integer to reduce overflows.
                (int) (((long) value * b.value) % prime),
                false
        );
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
        return value == 0;
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
        return value == 1;
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public PrimeFieldElement getZero() {
        return new PrimeFieldElement(prime, 0);
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public PrimeFieldElement getOne() {
        return new PrimeFieldElement(prime, 1);
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
    public int compareTo(PrimeFieldElement b) {
        ensureSameField(this, b);
        return Integer.compare(this.value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return (double) value;
    }


    @Override
    public int characteristic() {
        return prime;
    }


    @Override
    public int degree() {
        return 1;
    }
}
