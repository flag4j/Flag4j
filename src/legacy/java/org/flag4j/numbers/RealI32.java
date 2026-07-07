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

// TODO: Improve javadoc.
/**
 * <p>A real number backed by a 32-bit integer number. Immutable.
 *
 * <p>This class wraps the primitive int type.
 */
public class RealI32 implements RingElement<RealI32> {
    private static final long serialVersionUID = 1L;

    // Constants provided for convenience.
    /**
     * The numerical value -1.
     */
    public final static RealI32 NEGATIVE_ONE = new RealI32(-1);
    /**
     * The numerical value 0.
     */
    public final static RealI32 ZERO = new RealI32(0);
    /**
     * The numerical value 1.
     */
    public final static RealI32 ONE = new RealI32(1);
    /**
     * The numerical value 2.
     */
    public final static RealI32 TWO = new RealI32(2);
    /**
     * The numerical value 4.
     */
    public final static RealI32 THREE = new RealI32(3);
    /**
     * The numerical value 10.
     */
    public final static RealI32 TEN = new RealI32(10);


    /**
     * Numerical value of this ring element.
     */
    private final int value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the integer number.
     */
    public RealI32(int value) {
        this.value = value;
    }


    /**
     * Constructs a real 32-bit integer number from a {@code RealI16}.
     * @param value Value of the 32-bit integer number.
     */
    public RealI32(RealI16 value) {
        this.value = value.getValue();
    }


    /**
     * Gets the value of this ring element.
     * @return The value of this ring element.
     */
    public float getValue() {
        return value;
    }


    /**
     * Sums two elements of this ring (associative and commutative).
     *
     * @param b Second ring element in the sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealI32 add(RealI32 b) {
        return new RealI32(value + b.value);
    }


    /**
     * Adds an integer to this ring element.
     * @param b The integer to add to this ring element.
     * @return
     */
    public RealI32 add(int b) {
        return new RealI32(value + b);
    }


    /**
     * Computes the difference of two elements in this ring.
     *
     * @param b Second ring element in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public RealI32 sub(RealI32 b) {
        return new RealI32(value - b.value);
    }


    /**
     * Computes the difference of this ring element and an integer.
     *
     * @param b Integer to subtract from this ring element.
     *
     * @return The difference of this ring element and {@code b}.
     */
    public RealI32 sub(int b) {
        return new RealI32(value - b);
    }


    /**
     * Multiplies two elements of this ring (associative and commutative).
     *
     * @param b Second ring element in the product.
     *
     * @return The product of this ring element and {@code b}.
     */
    @Override
    public RealI32 mult(RealI32 b) {
        return new RealI32(value * b.value);
    }


    /**
     * Multiplies this ring element with an integer.
     *
     * @param b Integer to multiply this ring element with.
     *
     * @return The product of this ring element and {@code b}.
     */
    public RealI32 mult(int b) {
        return new RealI32(value * b);
    }


    /**
     * <p>Checks if this value is an additive identity for this ring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the ring.
     *
     * @return True if this value is an additive identity for this ring. Otherwise, false.
     */
    @Override
    public boolean isZero() {
        return equals(ZERO);
    }


    /**
     * <p>Checks if this value is a multiplicative identity for this ring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the ring.
     *
     * @return True if this value is a multiplicative identity for this ring. Otherwise, false.
     */
    @Override
    public boolean isOne() {
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this ring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the ring.
     *
     * @return The additive identity for this ring.
     */
    @Override
    public RealI32 getZero() {
        return ZERO;
    }


    /**
     * <p>Gets the multiplicative identity for this ring.
     *
     * <p>An element 1 is a multiplicative identity if a * 1 = a for any a in the ring.
     *
     * @return The multiplicative identity for this ring.
     */
    @Override
    public RealI32 getOne() {
        return ONE;
    }


    /**
     * <p>Computes the additive inverse for an element of this ring.
     *
     * <p>An element -x is an additive inverse for a filed element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this ring element.
     */
    @Override
    public RealI32 negate() {
        return new RealI32(-this.value);
    }


    /**
     * Computes the magnitude of this ring element.
     *
     * @return The magnitude of this ring element.
     */
    @Override
    public double mag() {
        return Math.abs(this.value);
    }


    /**
     * Evaluates the signum or sign function on a ring element.
     *
     * @param a Value to evaluate signum function on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealI32 sgn(RealI32 a) {
        if(a.value == 0) return a;
        else return (a.value > 0) ? ONE : NEGATIVE_ONE;
    }


    /**
     * Compares this element of the ordered ring with {@code b}.
     *
     * @param b Second element of the ordered ring.
     *
     * @return An int value:
     * <ul>
     *     <li>0 if this ring element is equal to {@code b}.</li>
     *     <li>< 0 if this ring element is less than {@code b}.</li>
     *     <li>> 0 if this ring element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two ring elements are equal, a negative value if and only the ring
     *     element it was called on is less than {@code b} and positive if and only if the ring element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    public int compareTo(RealI32 b) {
        return Integer.compare(value, b.value);
    }


    /**
     * Converts this ring value to an equivalent double value.
     *
     * @return A double value equivalent to this ring element.
     */
    @Override
    public double doubleValue() {
        return value;
    }


    /**
     * Checks if an object is equal to this ring element.
     * @param b Object to compare to this ring element.
     * @return True if the objects are the same or are both {@link RealI32}s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealI32) b).value;
    }


    @Override
    public int hashCode() {
        return Integer.hashCode(value);
    }


    /**
     * Converts this ring element to a string representation.
     * @return A string representation of this ring element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
