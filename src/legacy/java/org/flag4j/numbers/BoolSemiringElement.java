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


/**
 * Represents an immutable boolean value within a semiring structure.
 *
 * <p>This class wraps a primitive {@code boolean} value and provides operations consistent with a boolean semiring.
 * This structure is useful for representing boolean algebra operations. This is similar to the {@link BinaryFieldElement} class,
 * but addition is defined as logical OR in this class.
 *
 * <p>In this semiring:
 * <ul>
 *   <li><b>Addition</b> is defined as logical OR ({@code a + b = a OR b = a || b}).</li>
 *   <li><b>Multiplication</b> is defined as logical AND ({@code a * b = a AND b = a && b}).</li>
 *   <li><b>Additive Identity</b> is {@code false} (denoted by {@link #ZERO} or {@link #FALSE}).</li>
 *   <li><b>Multiplicative Identity</b> is {@code true} (denoted by {@link #ONE} or {@link #TRUE}).</li>
 * </ul>
 *
 * <p>The class implements the {@link SemiringElement} interface and adheres to its contract.
 * Instances of {@code BoolSemiringElement} are immutable and thread-safe.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * BoolSemiringElement a = BoolSemiringElement.TRUE;  // Equivalent to new BoolSemiringElement(true)
 * BoolSemiringElement b = BoolSemiringElement.FALSE; // Equivalent to new BoolSemiringElement(false)
 *
 * BoolSemiringElement sum = a.add(b);       // Logical OR: true || false => true
 * BoolSemiringElement product = a.mult(b);  // Logical AND: true && false => false
 * BoolSemiringElement notA = a.not();       // Logical NOT: !true => false
 * }</pre>
 *
 * <h2>Constants:</h2>
 * <p>The class provides constants for common boolean values:
 * <ul>
 *   <li>{@link #TRUE} or {@link #ONE} - Represents the boolean value {@code true}.</li>
 *   <li>{@link #FALSE} or {@link #ZERO} - Represents the boolean value {@code false}.</li>
 * </ul>
 *
 * @see SemiringElement
 * @see BinaryFieldElement
 */
public class BoolSemiringElement implements SemiringElement<BoolSemiringElement> {
    private static final long serialVersionUID = 1L;

    // Constants provided for convenience.
    /**
     * The boolean value true.
     */
    final public static BoolSemiringElement ONE = new BoolSemiringElement(true);
    /**
     * The boolean value true.
     */
    final public static BoolSemiringElement TRUE = ONE;
    /**
     * The boolean value false.
     */
    final public static BoolSemiringElement ZERO = new BoolSemiringElement(false);
    /**
     * The boolean value false.
     */
    final public static BoolSemiringElement FALSE = ZERO;


    /**
     * Boolean value of field element.
     */
    private final boolean value;


    /**
     * Constructs a {@code BoolSemiringElement} semiring element with the specified boolean value.
     *
     * @param value the boolean value to wrap.
     */
    public BoolSemiringElement(boolean value) {
        this.value = value;
    }


    /**
     * Constructs a {@code BoolSemiringElement} semiring element from an integer.
     *
     * @param value the integer value (must be 0 or 1).
     * @throws IllegalArgumentException if the value is not 0 or 1.
     */
    public BoolSemiringElement(int value) {
        if(value == 0) this.value = false;
        else if(value == 1) this.value = true;
        else throw new IllegalArgumentException("Cannot convert int value " + value + " to boolean. Must be 1 or 0.");
    }


    /**
     * <p>Performs the addition operation of this semiring, defined as logical OR.
     *
     * <p>This operation is associative and commutative.
     *
     * @param b the second semiring element to add.
     * @return A new {@code BoolSemiringElement} representing the logical OR of this value and {@code b}.
     */
    @Override
    public BoolSemiringElement add(BoolSemiringElement b) {
        return value || b.value ? TRUE : FALSE;
    }


    /**
     * <p>Computes the logical OR of this {@code BoolSemiringElement} with another.
     *
     * <p>This method is equivalent to {@link #add(BoolSemiringElement)}.
     *
     * @param b the {@code BoolSemiringElement} to perform the logical OR with.
     * @return A new {@code BoolSemiringElement} representing the logical OR of this value and {@code b}.
     */
    public BoolSemiringElement or(BoolSemiringElement b) {
        return value || b.value ? TRUE : FALSE;
    }


    /**
     * Computes the exclusive OR (XOR) of this {@code BoolSemiringElement} with another.
     *
     * @param b the {@code BoolSemiringElement} to perform the XOR with.
     * @return A new {@code BoolSemiringElement} representing the XOR of this value and {@code b}.
     */
    public BoolSemiringElement xor(BoolSemiringElement b) {
        return value ^ b.value ? TRUE : FALSE;
    }


    /**
     * <p>Performs the multiplication operation of this semiring, defined as logical AND.
     *
     * <p>This operation is associative.
     *
     * @param b the second semiring element to multiply.
     * @return A new {@code BoolSemiringElement} representing the logical AND of this value and {@code b}.
     */
    @Override
    public BoolSemiringElement mult(BoolSemiringElement b) {
        return value && b.value ? TRUE : FALSE;
    }


    /**
     * <p>Computes the logical AND of this {@code BoolSemiringElement} with another.
     *
     * <p>This method is equivalent to {@link #mult(BoolSemiringElement)}.
     *
     * @param b the {@code BoolSemiringElement} to perform the logical AND with.
     * @return A new {@code BoolSemiringElement} representing the logical AND of this value and {@code b}.
     */
    public BoolSemiringElement and(BoolSemiringElement b) {
        return mult(b);
    }


    /**
     * Computes the logical NOT (negation) of this {@code BoolSemiringElement}.
     *
     * @return a new {@code BoolSemiringElement} representing the logical NOT of this value.
     */
    public BoolSemiringElement not() {
        return !value ? TRUE : FALSE;
    }


    /**
     * <p>Checks if this value is an additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return {@code true} if this value is an additive identity for this semiring; {@code false} otherwise.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return {@code true} if this value is a multiplicative identity for this semiring; {@code false} otherwise.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * Gets the value of this semiring element.
     * @return The value of this semiring element.
     */
    public boolean getValue() {
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
    public BoolSemiringElement getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this semiring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the semiring.
     *
     * @return The multiplicative identity for this semiring.
     */
    @Override
    public BoolSemiringElement getOne() {
        return ONE;
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
    public int compareTo(BoolSemiringElement b) {
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
     * Checks if an object is equal to this semiring element.
     * @param b Object to compare to this semiring element.
     * @return True if the objects are the same or are both {@link BoolSemiringElement}s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((BoolSemiringElement) b).value;
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
