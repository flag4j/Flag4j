package org.flag4j.arrays.dense.semiring_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.SemiringMatrix;
import org.flag4j.io.PrintOptions;
import org.flag4j.numbers.Complex128;
import org.flag4j.numbers.SemiringElement;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class SemiringElementToStringTests {

    static SemiringElement<?>[] data;
    static SemiringElement<?>[][] data2D;
    static Shape shape;
    static SemiringMatrix<?> matrix;
    static String exp;


    @AfterEach
    void afterEach() {
        // Ensure all print options are reset.
        PrintOptions.resetAll();
    }


    @Test
    void complexToStringTests() {
        // ------------------ sub-case 1 ------------------
        data = new Complex128[]{new Complex128(1.3), new Complex128(1.4, -2),
                new Complex128(1.5, 9), new Complex128(-1, 2)};
        shape = new Shape(2, 2);
        matrix = new SemiringMatrix<Complex128>(shape, (Complex128[]) data);

        exp = "shape: (2, 2)\n" +
                "[[  1.3    1.4-2i ]\n" +
                " [ 1.5+9i  -1+2i  ]]";
        assertEquals(exp, matrix.toString());

        PrintOptions.setCentering(false);
        PrintOptions.setPadding(5);
        exp = "shape: (2, 2)\n" +
                "[[1.3        1.4-2i     ]\n" +
                " [1.5+9i     -1+2i      ]]";
        assertEquals(exp, matrix.toString());

        // ------------------ sub-case 2 ------------------
        PrintOptions.setCentering(true);
        PrintOptions.setPadding(3);
        shape = new Shape(22, 14);
        matrix = new SemiringMatrix<>(shape, new Complex128(-0.000001451, 21.22345));

        exp = "shape: (22, 14)\n" +
                "[[ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]\n" +
                " [                                                                                                          ...                                                                                                           ]\n" +
                " [ -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   -1.45E-6+21.22345i   ...   -1.45E-6+21.22345i  ]]";
        assertEquals(exp, matrix.toString());

        PrintOptions.setPrecision(3);
        PrintOptions.setMaxRows(5);
        PrintOptions.setMaxColumns(3);
        exp = "shape: (22, 14)\n" +
                "[[ 21.223i   21.223i   ...   21.223i  ]\n" +
                " [ 21.223i   21.223i   ...   21.223i  ]\n" +
                " [ 21.223i   21.223i   ...   21.223i  ]\n" +
                " [ 21.223i   21.223i   ...   21.223i  ]\n" +
                " [                ...                 ]\n" +
                " [ 21.223i   21.223i   ...   21.223i  ]]";
        assertEquals(exp, matrix.toString());
    }
}
