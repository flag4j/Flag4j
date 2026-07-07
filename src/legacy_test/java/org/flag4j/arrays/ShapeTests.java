package org.flag4j.arrays;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ShapeTests {
    int expRank, expValue;
    int[] expDims1, expDims2, expStrides, indices;
    Shape shape1;
    Shape shape2;


    @Test
    void testDimsConstructorTestCase() {
        // ----------- sub-case 1 -----------
        expRank = 3;
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);

        assertEquals(expRank, shape1.getRank());
        assertArrayEquals(expDims1, shape1.getDims());

        // ----------- sub-case 2 -----------
        expRank = 1;
        expDims1 = new int[]{8};
        shape1 = new Shape(expDims1);

        assertEquals(expRank, shape1.getRank());
        assertArrayEquals(expDims1, shape1.getDims());

        // ----------- sub-case 3 -----------
        expRank = 13;
        expDims1 = new int[]{8, 1, 2, 3, 4, 1, 10, 19, 304, 11, 8, 3, 90};
        shape1 = new Shape(expDims1);

        // ----------- sub-case 4 -----------
        expDims1 = new int[]{1, 3, -1, 0};
        assertThrows(IllegalArgumentException.class, () -> new Shape(expDims1));
    }


    @Test
    void getSizeTestCase() {
        // ----------- sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);

        for(int i = 0; i< expDims1.length; i++) {
            assertEquals(expDims1[i], shape1.getSize(i));
        }

        // ----------- sub-case 3 -----------
        expDims1 = new int[]{4, 903, 11, 45};
        shape1 = new Shape(expDims1);

        for(int i = 0; i< expDims1.length; i++) {
            assertEquals(expDims1[i], shape1.getSize(i));
        }
    }


    @Test
    void numelTestCase() {
        // ----------- sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);
        assertEquals(5 * 18, shape1.numel().intValue());

        // ----------- sub-case 2 -----------
        expDims1 = new int[]{4, 903, 11, 45};
        shape1 = new Shape(expDims1);
        assertEquals(4*903*11*45, shape1.numel().intValue());

        // ----------- sub-case 3 -----------
        expDims1 = new int[]{};
        shape1 = new Shape(expDims1); // Represents a scalar value.
        assertEquals(1, shape1.numel().intValue());
    }


    @Test
    void equalsTestCase() {
        // ----------- sub-case 1 -----------
        expDims1 = new int[]{1, 5, 18};
        expDims2 = new int[]{1, 5, 18};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertEquals(shape1, shape2);

        // ----------- sub-case 2 -----------
        expDims1 = new int[]{1, 5, 18};
        expDims2 = new int[]{1, 5, 18, 0};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertNotEquals(shape1, shape2);

        // ----------- sub-case 3 -----------
        expDims1 = new int[]{1};
        expDims2 = new int[]{4};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertNotEquals(shape1, shape2);

        // ----------- sub-case 4 -----------
        expDims1 = new int[]{};
        expDims2 = new int[]{};
        shape1 = new Shape(expDims1);
        shape2 = new Shape(expDims2);

        assertEquals(shape1, shape2);
    }


    @Test
    void strideTestCase() {
        // -------------- sub-case 1 --------------
        shape1 = new Shape(4, 2, 3);
        expStrides = new int[]{6, 3, 1};

        assertArrayEquals(expStrides, shape1.getStrides());

        // -------------- sub-case 2 --------------
        shape1 = new Shape();
        expStrides = new int[]{};

        assertArrayEquals(expStrides, shape1.getStrides());

        // -------------- sub-case 3 --------------
        shape1 = new Shape(15, 2, 3, 9);
        expStrides = new int[]{54, 27, 9, 1};

        assertArrayEquals(expStrides, shape1.getStrides());
    }

    @Test
    void entriesIndexTestCase() {
        // -------------- sub-case 1 --------------
        shape1 = new Shape(4, 2, 3);
        indices = new int[]{1, 0, 2};
        expValue = 8;
        assertEquals(expValue, shape1.get1DIndex(indices));

        // -------------- sub-case 2 --------------
        shape1 = new Shape(4, 2, 3);
        indices = new int[]{2, 1, 1};
        expValue = 16;
        assertEquals(expValue, shape1.get1DIndex(indices));

        // -------------- sub-case 3 --------------
        shape1 = new Shape(15, 2, 3, 9);
        indices = new int[]{11, 0, 1, 5};
        expValue = 608;
        assertEquals(expValue, shape1.get1DIndex(indices));

        // -------------- sub-case 4 --------------
        shape1 = new Shape(15, 2, 3, 9);
        indices = new int[]{11, 0, 1, 5, 1};
        assertThrows(IllegalArgumentException.class, () -> shape1.get1DIndex(indices));

        // -------------- sub-case 5 --------------
        shape1 = new Shape(15, 2, 3, 9);
        indices = new int[]{11, 2, 1, 5};
        assertThrows(IndexOutOfBoundsException.class, () -> shape1.get1DIndex(indices));

        // -------------- sub-case 6 --------------
        shape1 = new Shape(15, 2, 3, 9);
        indices = new int[]{11, 1, 1, 101};
        assertThrows(IndexOutOfBoundsException.class, () -> shape1.get1DIndex(indices));
    }


    @Test
    void swapAxesTestCase() {
        // -------------- sub-case 1 --------------
        shape1 = new Shape(4, 2, 3);
        shape2 = shape1.swapAxes(0, 1);
        expDims1 = new int[]{2, 4, 3};
        expStrides = new int[]{12, 3, 1};

        assertArrayEquals(expDims1, shape2.getDims());
        assertArrayEquals(expStrides, shape2.getStrides());

        // -------------- sub-case 2 --------------
        shape1 = new Shape(4, 2, 3);
        shape2 = shape1.swapAxes(1, 0);
        expDims1 = new int[]{2, 4, 3};
        expStrides = new int[]{12, 3, 1};

        assertArrayEquals(expDims1, shape2.getDims());
        assertArrayEquals(expStrides, shape2.getStrides());

        // -------------- sub-case 3 --------------
        shape1 = new Shape(4, 2, 3);
        shape2 = shape1.swapAxes(0, 2);
        expDims1 = new int[]{3, 2, 4};
        expStrides = new int[]{8, 4, 1};

        assertArrayEquals(expDims1, shape2.getDims());
        assertArrayEquals(expStrides, shape2.getStrides());
    }
}
