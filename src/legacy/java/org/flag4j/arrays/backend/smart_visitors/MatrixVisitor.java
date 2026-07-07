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

package org.flag4j.arrays.backend.smart_visitors;

import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.*;
import org.flag4j.arrays.sparse.*;
import org.flag4j.numbers.FieldElement;
import org.flag4j.numbers.RingElement;
import org.flag4j.numbers.SemiringElement;

// TODO: Docs
public abstract class MatrixVisitor<T> {

    protected final T other;

    public MatrixVisitor(T other) {
        this.other = other;
    }


    protected <U> String getInvalidOppMessage(MatrixMixin<?, ?, ?, ?> matrix) {
        return "Operation is not supported for matrix/vector types: "
                + matrix.getClass().getSimpleName() + " and " + other.getClass().getSimpleName();
    }


    public abstract T visit(Matrix matrix);
    public abstract T visit(CooMatrix matrix);
    public abstract T visit(CsrMatrix matrix);


    public abstract T visit(CMatrix matrix);
    public abstract T visit(CooCMatrix matrix);
    public abstract T visit(CsrCMatrix matrix);


    public abstract <U extends FieldElement<U>> T visit(FieldMatrix<U> matrix);
    public abstract <U extends FieldElement<U>> T visit(CooFieldMatrix<U> matrix);
    public abstract <U extends FieldElement<U>> T visit(CsrFieldMatrix<U> matrix);


    public abstract <U extends SemiringElement<U>> T visit(SemiringMatrix<U> matrix);
    public abstract <U extends SemiringElement<U>> T visit(CooSemiringMatrix<U> matrix);
    public abstract <U extends SemiringElement<U>> T visit(CsrSemiringMatrix<U> matrix);


    public abstract <U extends RingElement<U>> T visit(RingMatrix<U> matrix);
    public abstract <U extends RingElement<U>> T visit(CooRingMatrix<U> matrix);
    public abstract <U extends RingElement<U>> T visit(CsrRingMatrix<U> matrix);
}
