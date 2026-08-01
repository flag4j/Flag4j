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

package org.flag4jv3.ndarrays;

import java.util.Arrays;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Element-type coordinate of an ArrayType. Hash-key friendly by construction:
 * built-ins are enum singletons, generics are records.
 */
public sealed interface ElemType permits ElemType.Builtin, ElemType.Generic {

    /**
     * Weakest -> strongest. Gates op legality: sub needs RING, div needs FIELD.
     */
    enum Algebra {
        SEMIRING, RING, FIELD;


        public boolean atLeast(Algebra o) {
            return ordinal() >= o.ordinal();
        }


        static Algebra max(Algebra a, Algebra b) {
            return a.atLeast(b) ? a : b;
        }
    }

    Algebra algebra();

    /**
     * Boxed element class: Double.class, DoublePair.class, Rational.class, ...
     */
    Class<?> elementClass();


    /**
     * One constant per specialized backing type.
     */
    enum Builtin implements ElemType {
        BOOL(Algebra.SEMIRING, Boolean.class),
        INT32(Algebra.RING, Integer.class),
        INT64(Algebra.RING, Long.class),
        FLT32(Algebra.FIELD, Float.class),
        FLT64(Algebra.FIELD, Double.class);
        // todo now: Implement the field, rings, semirings, etc.
//        CPLX64 (Algebra.FIELD,    Complex64.class),
//        CPLX128(Algebra.FIELD,    DoublePair.class);

        private final Algebra algebra;
        private final Class<?> elementClass;


        Builtin(Algebra algebra, Class<?> elementClass) {
            this.algebra = algebra;
            this.elementClass = elementClass;
        }


        @Override
        public Algebra algebra() {
            return algebra;
        }


        @Override
        public Class<?> elementClass() {
            return elementClass;
        }


        private static final Map<Class<?>, Builtin> BY_CLASS = Arrays.stream(values())
                .collect(Collectors.toUnmodifiableMap(Builtin::elementClass, b -> b));


        static Builtin forClass(Class<?> c) {
            return BY_CLASS.get(c);
        }


        /**
         * Symmetric promotion table; null = no join. This table is POLICY.
         */
        private static final Builtin[][] JOIN = new Builtin[values().length][values().length];

        static {
            for (Builtin t : values()) def(t, t, t);              // idempotence

            def(INT32, INT64, INT64);

            // Policy: any int mixed with any float/complex promotes to the 64-bit variant.
            def(INT32, FLT32, FLT64);
            def(INT64, FLT32, FLT64);
            def(INT32, FLT64, FLT64);
            def(INT64, FLT64, FLT64);
//            def(INT32, CPLX64,  CPLX128); def(INT64, CPLX64,  CPLX128);
//            def(INT32, CPLX128, CPLX128); def(INT64, CPLX128, CPLX128);

            def(FLT32, FLT64, FLT64);
//            def(FLT32, CPLX64,  CPLX64);
//            def(FLT32, CPLX128, CPLX128);
//            def(FLT64, CPLX64,  CPLX128); // 64-bit real parts need 128-bit complex
//            def(FLT64, CPLX128, CPLX128);
//            def(CPLX64, CPLX128, CPLX128);

            // Deliberately absent: BOOL with anything. See discussion.
        }

        private static void def(Builtin a, Builtin b, Builtin r) {
            JOIN[a.ordinal()][b.ordinal()] = r;
            JOIN[b.ordinal()][a.ordinal()] = r; // symmetry by construction
        }
    }


    /**
     * Semiring/Ring/Field containers over a user-supplied element class.
     */
    record Generic(Algebra algebra, Class<?> elementClass) implements ElemType {
        public Generic {
            Objects.requireNonNull(algebra);
            Objects.requireNonNull(elementClass);
        }
    }


    /**
     * Descriptor factory; canonicalizes classes that have specialized backings.
     */
    static ElemType of(Algebra algebra, Class<?> elementClass) {
        Builtin b = Builtin.forClass(elementClass);
        return b != null ? b : new Generic(algebra, elementClass);
    }


    /**
     * Least upper bound in the element lattice; empty if the types don't combine.
     */
    static Optional<ElemType> join(ElemType a, ElemType b) {
        if (a.equals(b)) return Optional.of(a);

        if (a instanceof Builtin x && b instanceof Builtin y) {
            return Optional.ofNullable(Builtin.JOIN[x.ordinal()][y.ordinal()]);
        }

        // Same element class held at different algebra levels, e.g.,
        // RingMatrix<Rational> ⊔ FieldMatrix<Rational>. The FIELD-level container
        // could only be constructed if the class implements Field, so joining
        // upward to the stronger container is always safe.
        if (a instanceof Generic(Algebra algebra, Class<?> elementClass)
                && b instanceof Generic(Algebra algebra1, Class<?> aClass)
                && elementClass == aClass) {
            return Optional.of(new Generic(Algebra.max(algebra, algebra1), elementClass));
        }

        return Optional.empty();
    }
}
