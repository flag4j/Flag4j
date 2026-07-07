package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorPropertiesTests {

    Shape shape;
    double[] entries;
    Tensor A;


    @Test
    void isZerosTestCase() {
        // ----------------------- sub-case 1 -----------------------
        entries = new double[1024*4*9];
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllZeros());

        // ----------------------- sub-case 2 -----------------------
        entries = new double[1024*4*9];
        entries[1345] = 1.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllZeros());

        // ----------------------- sub-case 3 -----------------------
        entries = new double[]{0, 0.0, -0.0};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllZeros());

        // ----------------------- sub-case 4 -----------------------
        entries = new double[]{0, 0.0, -0.0, 1.34};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllZeros());
    }


    @Test
    void isOnesTestCase() {
        // ----------------------- sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllOnes());

        // ----------------------- sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1);
        entries[1345] = -131.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllOnes());

        // ----------------------- sub-case 3 -----------------------
        entries = new double[]{1, 1.0, 1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllOnes());

        // ----------------------- sub-case 4 -----------------------
        entries = new double[]{1, 1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllOnes());
    }


    @Test
    void isPosTestCase() {
        // ----------------------- sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 24.0);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllPos());

        // ----------------------- sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, 1515.11331);
        entries[1345] = -131.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllPos());

        // ----------------------- sub-case 3 -----------------------
        entries = new double[]{144, 1, 1.31415, 512.234345};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllPos());

        // ----------------------- sub-case 4 -----------------------
        entries = new double[]{1, 1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllPos());
    }


    @Test
    void isNegTestCase() {
        // ----------------------- sub-case 1 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, -24.0);
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllNeg());

        // ----------------------- sub-case 2 -----------------------
        entries = new double[1024*4*9];
        Arrays.fill(entries, -1515.11331);
        entries[1345] = 1.134;
        shape = new Shape(1, 1024, 1, 9, 4, 1, 1);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllNeg());

        // ----------------------- sub-case 3 -----------------------
        entries = new double[]{-144, -1, -1.31415, -512.234345};
        shape = new Shape(4);
        A = new Tensor(shape, entries);

        assertTrue(A.isAllNeg());

        // ----------------------- sub-case 4 -----------------------
        entries = new double[]{1, -1.0, -1};
        shape = new Shape(3);
        A = new Tensor(shape, entries);

        assertFalse(A.isAllNeg());
    }
}
