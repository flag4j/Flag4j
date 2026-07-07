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

public class AddVisitor extends MatrixVisitor<MatrixMixin<?, ?, ?, ?>> {

    public AddVisitor(MatrixMixin<?, ?, ?, ?> other) {
        super(other);
    }

    @Override
    public MatrixMixin<?, ?, ?, ?> visit(Matrix matrix) {
        if(other instanceof Matrix)
            return matrix.add((Matrix) other);
        else if(other instanceof CMatrix)
            return matrix.add((CMatrix) other);
        else if(other instanceof CooMatrix)
            return matrix.add((CooMatrix) other);
        else if(other instanceof CooCMatrix)
            return matrix.add((CooCMatrix) other);
        else if(other instanceof CsrMatrix)
            return matrix.add((CsrMatrix) other);
        else if(other instanceof CsrCMatrix)
            return matrix.add((CsrCMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CMatrix matrix) {
        if(other instanceof Matrix)
            return matrix.add((Matrix) other);
        else if(other instanceof CMatrix)
            return matrix.add((CMatrix) other);
        else if(other instanceof CooMatrix)
            return matrix.add((CooMatrix) other);
        else if(other instanceof CooCMatrix)
            return matrix.add((CooCMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CooMatrix matrix) {
        if(other instanceof Matrix)
            return ((Matrix) other).add(matrix);
        else if(other instanceof CMatrix)
            return ((CMatrix) other).add(matrix);
        else if(other instanceof CooMatrix)
            return matrix.add((CooMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CooCMatrix matrix) {
        if(other instanceof Matrix)
            return ((Matrix) other).add(matrix);
        else if(other instanceof CMatrix)
            return ((CMatrix) other).add(matrix);
        else if(other instanceof CooCMatrix)
            return matrix.add((CooCMatrix) other);


        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CsrMatrix matrix) {
        if(other instanceof CsrMatrix)
            return matrix.add((CsrMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public MatrixMixin<?, ?, ?, ?> visit(CsrCMatrix matrix) {
        if(other instanceof CsrCMatrix)
            return matrix.add((CsrCMatrix) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends FieldElement<U>> MatrixMixin<?, ?, ?, ?> visit(FieldMatrix<U> matrix) {
        if(other instanceof FieldMatrix<?>)
            return matrix.add((FieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends FieldElement<U>> MatrixMixin<?, ?, ?, ?> visit(CooFieldMatrix<U> matrix) {
        if(other instanceof CooFieldMatrix<?>)
            return matrix.add((CooFieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends FieldElement<U>> MatrixMixin<?, ?, ?, ?> visit(CsrFieldMatrix<U> matrix) {
        if(other instanceof CsrFieldMatrix<?>)
            return matrix.add((CsrFieldMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends SemiringElement<U>> MatrixMixin<?, ?, ?, ?> visit(SemiringMatrix<U> matrix) {
        if(other instanceof SemiringMatrix<?>)
            return matrix.add((SemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends SemiringElement<U>> MatrixMixin<?, ?, ?, ?> visit(CooSemiringMatrix<U> matrix) {
        if(other instanceof CooSemiringMatrix<?>)
            return matrix.add((CooSemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends SemiringElement<U>> MatrixMixin<?, ?, ?, ?> visit(CsrSemiringMatrix<U> matrix) {
        if(other instanceof CsrSemiringMatrix<?>)
            return matrix.add((CsrSemiringMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends RingElement<U>> MatrixMixin<?, ?, ?, ?> visit(RingMatrix<U> matrix) {
        if(other instanceof RingMatrix<?>)
            return matrix.add((RingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends RingElement<U>> MatrixMixin<?, ?, ?, ?> visit(CooRingMatrix<U> matrix) {
        if(other instanceof CooRingMatrix<?>)
            return matrix.add((CooRingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }


    @Override
    public <U extends RingElement<U>> MatrixMixin<?, ?, ?, ?> visit(CsrRingMatrix<U> matrix) {
        if(other instanceof CsrRingMatrix<?>)
            return matrix.add((CsrRingMatrix<U>) other);

        throw new UnsupportedOperationException(getInvalidOppMessage(matrix));
    }
}
