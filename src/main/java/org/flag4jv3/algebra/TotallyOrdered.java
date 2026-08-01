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

package org.flag4jv3.algebra;

/// Represents an element of a set which is totally orderable.
public interface TotallyOrdered<T extends TotallyOrdered<T>> extends Comparable<T> {


    /// Compares this element of the with `b`.
    ///
    /// @param b Second element.
    /// @return An int value:
    /// - `0` if `this` element is equal to `b`.
    /// - `< 0` if `this` element is less than `b`.
    /// - `> 0` if `this` element is greater than `b`.
    ///
    /// Hence, this method returns zero if and only if the two elements are equal, a negative value if and only the
    /// element it was called on is less than {@code b} and positive if and only if theelement it was called on is greater
    /// than {@code b}.
    @Override
    int compareTo(T b);
}
