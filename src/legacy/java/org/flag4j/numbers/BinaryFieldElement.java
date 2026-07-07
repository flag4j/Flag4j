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
 * Represents an immutable boolean value within a field structure. Specifically, this is the finite field GF(2).
 *
 * <p>This class wraps a primitive {@code boolean} value and provides operations consistent with a boolean field.
 *
 * <p>In this field:
 * <ul>
 *   <li><b>Addition</b> is defined as exclusive OR ({@code a + b = a XOR b = a ^ b).</li>
 *   <li><b>Multiplication</b> is defined as logical AND ({@code a * b = a AND b = a && b}).</li>
 *   <li><b>Additive Identity</b> is {@code false} (denoted by {@link #ZERO} or {@link #FALSE}).</li>
 *   <li><b>Multiplicative Identity</b> is {@code true} (denoted by {@link #ONE} or {@link #TRUE}).</li>
 * </ul>
 *
 * <p>The class implements the {@link FieldElement } interface and adheres to its contract.
 * Instances of {@code BinaryFieldElement} are immutable and thread-safe.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * BinaryFieldElement a = BinaryFieldElement.TRUE;  // Equivalent to new BinaryFieldElement(true)
 * BinaryFieldElement b = BinaryFieldElement.FALSE; // Equivalent to new BinaryFieldElement(false)
 *
 * BinaryFieldElement sum = a.add(b);       // Exclusive OR: true ^ false => true
 * BinaryFieldElement product = a.mult(b);  // Logical AND: true && false => false
 * BinaryFieldElement notA = a.not();       // Logical NOT: !true => false
 * }</pre>
 *
 * <h2>Constants:</h2>
 * <p>The class provides constants for common boolean values:
 * <ul>
 *   <li>{@link #TRUE} or {@link #ONE} - Represents the boolean value {@code true}.</li>
 *   <li>{@link #FALSE} or {@link #ZERO} - Represents the boolean value {@code false}.</li>
 * </ul>
 *
 * @see FieldElement
 * @see BoolSemiringElement
 */
public class BinaryFieldElement extends FiniteFieldElement<BinaryFieldElement> {
    private static final long serialVersionUID = 1L;

    // Constants provided for convenience.
    /**
     * The boolean value true.
     */
    final public static BinaryFieldElement ONE = new BinaryFieldElement(true);
    /**
     * The boolean value true.
     */
    final public static BinaryFieldElement TRUE = ONE;
    /**
     * The boolean value false.
     */
    final public static BinaryFieldElement ZERO = new BinaryFieldElement(false);
    /**
     * The boolean value false.
     */
    final public static BinaryFieldElement FALSE = ZERO;

    /**
     * Boolean value of field element.
     */
    private final boolean value;


    /**
     * Constructs a {@code BinaryFieldElement} field element with the specified boolean value.
     *
     * @param value the boolean value to wrap.
     */
    public BinaryFieldElement(boolean value) {
        this.value = value;
    }


    /**
     * Constructs a {@code BinaryFieldElement} field element from an integer.
     *
     * @param value the integer value (must be 0 or 1).
     * @throws IllegalArgumentException if the value is not 0 or 1.
     */
    public BinaryFieldElement(int value) {
        if(value == 0) this.value = false;
        else if(value == 1) this.value = true;
        else throw new IllegalArgumentException("Cannot convert int value " + value + " to boolean. Must be 1 or 0.");
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public boolean getValue() {
        return value;
    }


    /**
     * Computes the quotient of two elements of this field.
     *
     * @param b Second field element in quotient.
     *
     * @return The quotient of this field element and {@code b}.
     * @throws ArithmeticException If {@code b} is zero.
     */
    @Override
    public BinaryFieldElement div(BinaryFieldElement b) {
        if(!b.value) {
            throw new ArithmeticException("Division by zero.");
        }

        return this; // The only valid divisions are 0/1 = 0 and 1/1 = 1.
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
    public BinaryFieldElement invert() {
        if(!value) {
            throw new ArithmeticException("Zero has no multiplicative inverse.");
        }

        return this;
    }


    @Override
    public int characteristic() {
        return 2;
    }


    @Override
    public int degree() {
        return 1;
    }


    /**
     * <p>Checks if this field element is finite in magnitude.
     * <p>Note: A boolean field is a finite field, so this will @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.). return {@code true}.
     *
     * @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.).
     */
    @Override
    public boolean isFinite() {
        return true;
    }


    /**
     * <p>Checks if this field element is infinite in magnitude.
     * <p>Note: A boolean field is a finite field, so this will @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.). return true.
     * @return {@code true} if this field element is infinite in magnitude; {@code false} otherwise (i.e., finite, NaN, etc.).
     */
    @Override
    public boolean isInfinite() {
        return false;
    }


    /**
     * <p>Checks if this field element is NaN in magnitude.
     * <p>Note: A boolean field is a finite field, so this will @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.). return true.
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
    public BinaryFieldElement sub(BinaryFieldElement b) {
        return value ^ b.value ? TRUE : FALSE;
    }


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     * <p>Note: The additive inverse of an element from a boolean field is @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.). itself.
     *
     * @return The additive inverse for this ring element.
     */
    @Override
    public BinaryFieldElement negate() {
        return this;
    }


    /**
     * Sums two elements of this semiring (associative and commutative).
     *
     * @param b Second semiring element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public BinaryFieldElement add(BinaryFieldElement b) {
        return value ^ b.value ? TRUE : FALSE;
    }


    /**
     * Multiplies two elements of this semiring (associative).
     *
     * @param b Second semiring element in the product.
     *
     * @return The product of this semiring element and {@code b}.
     */
    @Override
    public BinaryFieldElement mult(BinaryFieldElement b) {
        return value && b.value ? TRUE : FALSE;
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
        return !value;
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
        return value;
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public BinaryFieldElement getZero() {
        return FALSE;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public BinaryFieldElement getOne() {
        return TRUE;
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
    public int compareTo(BinaryFieldElement b) {
        return Boolean.compare(value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return (value) ? 1.0 : 0.0;
    }


    /**
     * <p>Computes the logical OR of this {@code BinaryFieldElement} with another.
     *
     * @param b the {@code BoolFeild} to perform the logical OR with.
     * @return A new {@code BinaryFieldElement} representing the logical OR of this value and {@code b}.
     */
    public BinaryFieldElement or(BinaryFieldElement b) {
        return value || b.value ? TRUE : FALSE;
    }


    /**
     * <p>Computes the exclusive OR (XOR) of this {@code BinaryFieldElement} with another.
     * <p>This method is equivalent to {@link #add(BinaryFieldElement)}.
     *
     * @param b the {@code BinaryFieldElement} to perform the XOR with.
     * @return A new {@code BinaryFieldElement} representing the XOR of this value and {@code b}.
     */
    public BinaryFieldElement xor(BinaryFieldElement b) {
        return value ^ b.value ? TRUE : FALSE;
    }


    /**
     * <p>Computes the logical AND of this {@code BinaryFieldElement} with another.
     *
     * <p>This method is equivalent to {@link #mult(BinaryFieldElement)}.
     *
     * @param b the {@code BinaryFieldElement} to perform the logical AND with.
     * @return A new {@code BinaryFieldElement} representing the logical AND of this value and {@code b}.
     */
    public BinaryFieldElement and(BinaryFieldElement b) {
        return value && b.value ? TRUE : FALSE;
    }


    /**
     * Computes the logical NOT (negation) of this {@code BinaryFieldElement}.
     *
     * @return a new {@code BinaryFieldElement} representing the logical NOT of this value.
     */
    public BinaryFieldElement not() {
        return !value ? TRUE : FALSE;
    }


    /**
     * Checks if an object is equal to this field element.
     * @param b Object to compare to this field element.
     * @return True if the objects are the same or are both {@link BinaryFieldElement}s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((BinaryFieldElement) b).value;
    }


    @Override
    public int hashCode() {
        return Boolean.hashCode(value);
    }


    /**
     * Forms a human-readable string representing this {@link BoolSemiringElement} instance.
     * @return A human-readable string representing this {@link BoolSemiringElement} instance.
     */
    public String toString() {
        return Boolean.toString(value);
    }
}
