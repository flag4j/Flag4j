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

package org.flag4j.arrays.backend;

/**
 * This interface specifies methods which all vectors should implement.
 * @param <T> Type of this vector.
 * @param <U> Type of matrix which is similar to {@code T}.
 * @param <V> Type of dense matrix which is similar to {@code U}. If {@code T} is dense, then {@code U} and {@code T} should be the
 * same type.
 * @param <W> Type (or wrapper of) an individual element of the vector.
 */
public interface VectorMixin<T extends VectorMixin<T, U, V, W>,
        U extends MatrixMixin<U, V, T, W>, V extends MatrixMixin<V, V, ?, W>, W>  {

    /**
     * Joints specified vector with this vector. That is, creates a vector of length {@code this.length() + b.length()} containing
     * first the elements of this vector followed by the elements of {@code b}.
     *
     * @param b Vector to join with this vector.
     * @return A vector resulting from joining the specified vector with this vector.
     */
    T join(T b);


    /**
     * <p>Computes the inner product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #dot(VectorMixin)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the inner product.
     * @return The inner product between this vector and the vector {@code b}.
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #dot(VectorMixin)
     */
    W inner(T b);


    /**
     * <p>Computes the dot product between two vectors.
     *
     * <p>Note: this method is distinct from {@link #inner(VectorMixin)}. The inner product is equivalent to the dot product
     * of this tensor with the conjugation of {@code b}.
     *
     * @param b Second vector in the dot product.
     *
     * @return The dot product between this vector and the vector {@code b}.
     *
     * @throws IllegalArgumentException If this vector and vector {@code b} do not have the same number of data.
     * @see #inner(VectorMixin)
     */
    W dot(T b);


    /**
     * Gets the length of a vector. Same as {@link #size()}.
     *
     * @return The length, i.e., the number of data, in this vector.
     */
    int length();


    /**
     * Gets the size/length of a vector. Same as {@link #length()}.
     * @return The length, i.e., the number of data, in this vector.
     */
    default int size() {
        return length();
    }


    /**
     * Repeats a vector {@code n} times along a certain axis to create a matrix.
     *
     * @param n Number of times to repeat a vector.
     * @param axis Axis along which to repeat vector:
     * <ul>
     *     <li>If {@code axis=0}, then the vector will be treated as a row vector and stacked vertically {@code n} times.</li>
     *     <li>If {@code axis=1} then the vector will be treated as a column vector and stacked horizontally {@code n} times.</li>
     * </ul>
     *
     * @return A matrix whose rows/columns are this vector repeated.
     */
    U repeat(int n, int axis);


    /**
     * Stacks two vectors vertically as if they were row vectors to form a matrix with two rows.
     *
     * @param b Vector to stack below this vector.
     * @return The result of stacking this vector and vector {@code b}.
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of data in
     *                                  the vector {@code b}.
     */
    default U stack(T b) {
        return stack(b, 0);
    }


    /**
     * <p>
     * Stacks two vectors along the specified axis.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 0 stacks the vectors
     * as if they were row vectors resulting in a {@code 2&times;n} matrix.
     * 
     *
     * <p>
     * Stacking two vectors of length {@code n} along axis 1 stacks the vectors
     * as if they were column vectors resulting in a {@code n&times;2} matrix.
     * 
     *
     * @param b    Vector to stack with this vector.
     * @param axis Axis along which to stack vectors. If {@code axis=0}, then vectors are stacked as if they are row
     *             vectors. If {@code axis=1}, then vectors are stacked as if they are column vectors.
     * @return The result of stacking this vector and the vector {@code b}.
     * @throws IllegalArgumentException If the number of data in this vector is different from the number of
     *                                  data in the vector {@code b}.
     * @throws IllegalArgumentException If the axis is not either 0 or 1.
     */
    U stack(T b, int axis);


    /**
     * Computes the outer product of two vectors.
     *
     * @param b Second vector in the outer product.
     * @return The result of the vector outer product between this vector and {@code b}.
     * @throws IllegalArgumentException If the two vectors do not have the same number of data.
     */
    V outer(T b);


    /**
     * Converts a vector to an equivalent matrix representing the vector as a column vector.
     *
     * @return A matrix equivalent to this vector as if it were a column vector.
     */
    default U toMatrix() {
        return toMatrix(true);
    }


    /**
     * Converts a vector to an equivalent matrix representing either a row or column vector.
     * @param columVector Flag indicating whether to convert this vector to a matrix representing a row or column vector:
     *                    <p>If {@code true}, the vector will be converted to a matrix representing a column vector.
     *                    <p>If {@code false}, The vector will be converted to a matrix representing a row vector.
     * @return A matrix equivalent to this vector.
     */
    U toMatrix(boolean columVector);


    /**
     * Normalizes this vector to a unit length vector.
     * @return This vector normalized to a unit length.
     */
    T normalize();


    /**
     * Computes the magnitude of this vector.
     * @return The magnitude of this vector.
     */
    W mag();


    /**
     * Gets multiple items from this vector.
     * @param indices The indices of each item to get from this vector.
     * @return A vector containing the entries of this vector at the specified {@code indices}.
     * @throws IndexOutOfBoundsException If any index in {@code indices} is not within the bounds of this vector.
     * @see #get(int) 
     * @see #getSlice(int, int) 
     */
    VectorMixin<?, ?, ?, W> getItems(int... indices);


    /**
     * Gets the element of this vector at the specified index.
     * @param idx Index of the element to get within this vector.
     * @return The element of this vector at index {@code idx}.
     * @throws IndexOutOfBoundsException If {@code idx} is not within the bounds of this vector.
     * @see #getSlice(int, int)
     * @see #getItems(int...) 
     */
    W get(int idx);


    /**
     * Gets a slice of this vector over the specified range of indices.
     * @param startIdx Staring index of slice (inclusive).
     * @param endIdx Ending index of slice (exclusive).
     * @return A vector of length {@code endIdx - startIdx} whose entries are the elements of this vector
     * over the specified range of indices [{@code startIdx}, {@code endIdx}).
     * @throws IndexOutOfBoundsException If {@code startIdx} or {@code endIdx - 1} are not within the bounds of this vector.
     * @throws IllegalArgumentException If {@code startIdx > endIdx}.
     * @see #get(int)
     * @see #getItems(int...)
     */
    T getSlice(int startIdx, int endIdx);


    /**
     * Sets a slice of this vector to the entries of another vector.
     * @param values A vector containing the values to set.
     * @param startIdx The starting index of the slice to set. The size of the slice will be {@code values.length}.
     * @return If this vector is dense, the operation will be done in-place and a reference to this tensor will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and this copy will be returned.
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector,
     * or if {@code values} does not fit within this vector when its first entry is placed at {@code startIdx}.
     */
    T setSlice(T values, int startIdx);


    /**
     * Sets a slice of this vector to the entries of an array.
     * @param values Array containing the values to set.
     * @param startIdx The starting index of the slice to set. The size of the slice will be {@code values.length}.
     * @return If this vector is dense, the operation will be done in-place and a reference to this tensor will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and this copy will be returned.
     * @throws IndexOutOfBoundsException If {@code startIdx} is out of bounds of this vector,
     * or if {@code values} does not fit within this vector when its first entry is placed at {@code startIdx}.
     */
    T setSlice(W[] values, int startIdx);


    /**
     * Sets multiple items of this vector.
     * @param values New values it set the specified items to.
     * @param indices The indices indicating where each value in {@code values} should be set within this vector.
     * @return If this vector is dense, the operation will be done in-place and a reference to this vector will be returned.
     * If this vector is sparse, the operation will be done out-of-place in a copy of this vector, and that copy will be returned.
     * @throws IndexOutOfBoundsException If any index in {@code indices} is not within the bounds of this vector.
     * @throws IllegalArgumentException If {@code values.length != indices.length}.
     */
    T setItems(W[] values, int[] indices);
}
