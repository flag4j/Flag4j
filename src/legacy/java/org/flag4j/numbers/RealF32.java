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

import org.flag4j.util.ErrorMessages;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * <p>Represents an immutable real number backed by a 32-bit floating-point value.
 *
 * <p>This class wraps the primitive {@code float} type and provides arithmetic operations
 * consistent with a mathematical field. It implements the {@link FieldElement} interface,
 * allowing it to be used in generic algorithms that operate on field elements.
 *
 * @see RealF64
 */
public class RealF32 implements FieldElement<RealF32> {
    private static final long serialVersionUID = 1L;

    /**
     * The numerical value {@code -1.0}.
     */
    public final static RealF32 NEGATIVE_ONE = new RealF32(-1f);
    /**
     * The numerical value {@code 0.0}.
     */
    public final static RealF32 ZERO = new RealF32(0f);
    /**
     * The numerical value {@code 1.0}.
     */
    public final static RealF32 ONE = new RealF32(1f);
    /**
     * The numerical value {@code 2.0}.
     */
    public final static RealF32 TWO = new RealF32(2f);
    /**
     * The numerical value {@code 3.0}.
     */
    public final static RealF32 THREE = new RealF32(3f);
    /**
     * The numerical value {@code 10.0}.
     */
    public final static RealF32 TEN = new RealF32(10f);
    /**
     * The numerical value representing the square root of two.
     */
    public final static RealF32 ROOT_TWO = new RealF32((float) Math.sqrt(2d));
    /**
     * The numerical value representing the square root of three.
     */
    public final static RealF32 ROOT_THREE = new RealF32((float) Math.sqrt(3d));
    /**
     * The numerical value representing pi (π), the ratio of the circumference of a circle to its diameter.
     */
    public final static RealF32 PI = new RealF32((float) Math.PI);
    

    /**
     * Numerical value of field element.
     */
    private final float value;


    /**
     * Constructs a real 32-bit floating point number.
     * @param value Value of the 32-bit floating point number.
     */
    public RealF32(float value) {
        this.value = value;
    }


    /**
     * Constructs a real 32-bit floating point number by casting a real 64-bit floating point.
     * @param value Value of the 32-bit floating point number.
     */
    public RealF32(RealF64 value) {
        this.value = (float) value.getValue();
    }


    /**
     * Gets the value of this field element.
     * @return The value of this field element.
     */
    public float getValue() {
        return value;
    }


    /**
     * Sums two elements of this field (associative and commutative).
     *
     * @param b Second field element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealF32 add(RealF32 b) {
        return new RealF32(this.value + b.value);
    }


    /**
     * Computes difference of two elements of this field.
     *
     * @param b Second field element in difference.
     *
     * @return The difference of this field element and {@code b}.
     */
    @Override
    public RealF32 sub(RealF32 b) {
        return new RealF32(this.value - b.value);
    }


    /**
     * Multiplies two elements of this field (associative and commutative).
     *
     * @param b Second field element in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealF32 mult(RealF32 b) {
        return new RealF32(this.value * b.value);
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
        return equals(ZERO);
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
        return equals(ONE);
    }


    /**
     * <p>Gets the additive identity for this semiring.
     *
     * <p>An element 0 is an additive identity if a + 0 = a for any a in the semiring.
     *
     * @return The additive identity for this semiring.
     */
    @Override
    public RealF32 getZero() {
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
    public RealF32 getOne() {
        return ONE;
    }


    /**
     * Computes the quotient of two elements of this field.
     * @param b Second field element in quotient.
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public RealF32 div(RealF32 b) {
        return new RealF32(this.value / b.value);
    }


    /**
     * Sums an element of this field with a real number (associative and commutative).
     *
     * @param b Real element in sum.
     *
     * @return The sum of this element and {@code b}.
     */
    @Override
    public RealF32 add(double b) {
        return new RealF32((float) (value + b));
    }


    /**
     * Computes difference of an element of this field and a real number.
     *
     * @param b Real value in difference.
     *
     * @return The difference of this ring element and {@code b}.
     */
    @Override
    public RealF32 sub(double b) {
        return new RealF32((float) (value - b));
    }


    /**
     * Multiplies an element of this field with a real number (associative and commutative).
     *
     * @param b Real number in product.
     *
     * @return The product of this field element and {@code b}.
     */
    @Override
    public RealF32 mult(double b) {
        return new RealF32((float) (value * b));
    }


    /**
     * Computes the quotient of an element of this field and a real number.
     *
     * @param b Real number in quotient.
     *
     * @return The quotient of this field element and {@code b}.
     */
    @Override
    public RealF32 div(double b) {
        return new RealF32((float) (value / b));
    }


    /**
     * <p>Computes the multiplicative inverse for an element of this field.
     *
     * <p>An element x<sup>-1</sup> is a multiplicative inverse for a field element x if x<sup>-1</sup>*x = 1 where 1 is the
     * multiplicative identity.
     *
     * @return The multiplicative inverse for this field element.
     */
    @Override
    public RealF32 invert() {
        return new RealF32(1f/this.value);
    }


    /**
     * <p>Computes the additive inverse for an element of this field.
     *
     * <p>An element -x is an additive inverse for a field element x if -x + x = 0 where 0 is the additive identity.
     *
     * @return The additive inverse for this field element.
     */
    @Override
    public RealF32 negate() {
        return new RealF32(-this.value);
    }


    /**
     * Computes the magnitude of this field element.
     *
     * @return The magnitude of this field element.
     */
    @Override
    public double mag() {
        return Math.abs(this.value);
    }


    /**
     * Computes the square root of this field element.
     *
     * @return The square root of this field element.
     */
    @Override
    public RealF32 sqrt() {
        return new RealF32((float) Math.sqrt(this.value));
    }


    /**
     * Evaluates the signum or sign function on a field element.
     *
     * @param a Value to evaluate signum function on.
     * @return The output of the signum function evaluated on {@code a}.
     */
    public static RealF32 sgn(RealF32 a) {
        return new RealF32(Math.signum(a.value));
    }


    /**
     * Compares this element of the ordered field with {@code b}.
     *
     * @param b Second element of the ordered field.
     *
     * @return An int value:
     * <ul>
     *     <li>0 if this field element is equal to {@code b}.</li>
     *     <li>< 0 if this field element is less than {@code b}.</li>
     *     <li>> 0 if this field element is greater than {@code b}.</li>
     *     Hence, this method returns zero if and only if the two field elements are equal, a negative value if and only the field
     *     element it was called on is less than {@code b} and positive if and only if the field element it was called on is greater
     *     than {@code b}.
     * </ul>
     */
    @Override
    public int compareTo(RealF32 b) {
        return Float.compare(this.value, b.value);
    }


    /**
     * Converts this semiring value to an equivalent double value.
     *
     * @return A double value equivalent to this semiring element.
     */
    @Override
    public double doubleValue() {
        return value;
    }


    /**
     * Checks if this field element is finite in magnitude.
     *
     * @return {@code true} if this field element is finite in magnitude; {@code false} otherwise (i.e., infinite, NaN, etc.).
     */
    @Override
    public boolean isFinite() {
        return Float.isFinite(this.value);
    }


    /**
     * Checks if this field element is infinite in magnitude.
     *
     * @return {@code true} if this field element is infinite in magnitude; {@code false} otherwise (i.e., finite, NaN, etc.).
     */
    @Override
    public boolean isInfinite() {
        return Float.isInfinite(this.value);
    }


    /**
     * Checks if this field element is NaN in magnitude.
     *
     * @return {@code true} if this field element is NaN in magnitude; {@code false} otherwise (i.e., finite, NaN, etc.).
     */
    @Override
    public boolean isNaN() {
        return Float.isNaN(this.value);
    }


    /**
     * Computes the product of all items of specified array.
     * @param values Values to compute product of.
     * @return The product of all values in {@code values}.
     */
    public static RealF32 prod(RealF32... values) {
        if(values == null || values.length == 0)
            throw new IllegalArgumentException("Values cannot be null or empty");

        float prod = values[0].value;
        for(int i = 1, length=values.length; i < length; i++)
            prod *= values[i].value;

        return new RealF32(prod);
    }


    /**
     * Computes the sum of all items of specified array.
     * @param values Values to compute product of.
     * @return The sum of all values in {@code values}.
     */
    public static RealF32 sum(RealF32... values) {
        if(values == null || values.length == 0)
            throw new IllegalArgumentException("Values cannot be null or empty");

        float prod = values[0].value;
        for(int i = 1, length=values.length; i < length; i++)
            prod += values[i].value;

        return new RealF32(prod);
    }


    /**
     * Wraps a primitive float array as a {@link RealF32} array.
     * @param arr Array to wrap.
     * @return A {@link RealF32} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static RealF32[] wrapArray(float... arr) {
        if(arr == null) return null;

        RealF32[] wrapped = new RealF32[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new RealF32(arr[i]);

        return wrapped;
    }


    /**
     * Wraps a {@link Float} array as a {@link RealF32} array.
     * @param arr Array to wrap.
     * @return A {@link RealF32} array containing the values of {@code arr}. If {@code arr==null} then {@code null} will be
     * returned.
     */
    public static RealF32[] wrapArray(Float... arr) {
        if(arr == null) return null;

        RealF32[] wrapped = new RealF32[arr.length];
        for(int i=0, size=arr.length; i<size; i++)
            wrapped[i] = new RealF32(arr[i]);

        return wrapped;
    }


    /**
     * Rounds number to specified number of decimal places.
     *
     * @param n Number to round.
     * @param decimals Number of decimals to round to.
     * @return The number {@code n} rounded to the specified
     * 		number of decimals.
     * @throws IllegalArgumentException If decimals is less than zero.
     */
    public static RealF32 round(RealF32 n, int decimals) {
        if (decimals < 0)
            throw new IllegalArgumentException(ErrorMessages.getNegValueErr(decimals, "decimals"));

        float value = (Double.isFinite(n.value))
                ? BigDecimal.valueOf(n.value).setScale(decimals, RoundingMode.HALF_UP).floatValue()
                : n.value;

        return new RealF32(value);
    }


    /**
     * Checks if an object is equal to this FieldElement element.
     * @param b Object to compare to this FieldElement element.
     * @return True if the objects are the same or are both {@link RealF32}s and have equal values.
     */
    @Override
    public boolean equals(Object b) {
        // Check for quick returns.
        if(this == b) return true;
        if(b == null) return false;
        if(b.getClass() != this.getClass()) return false;

        return this.value == ((RealF32) b).value;
    }


    @Override
    public int hashCode() {
        return Float.hashCode(value);
    }


    /**
     * Converts this field element to a string representation.
     * @return A string representation of this field element.
     */
    public String toString() {
        return String.valueOf(this.value);
    }
}
