package org.flag4j.linalg.ops.dense.real;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RealDenseSetOpsTests {
    double[] dest, expDest;
    Double[] arrD;
    Integer[] arrI;
    double[] arrd;
    int[] arri;

    Double[][] arrDD;
    Integer[][] arrII;
    double[][] arrdd;
    int[][] arrii;


    @Test
    void oneDArrayTestCase() {
        // -------------- sub-case 1 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[arrD.length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOps.setValues(arrD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 2 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new double[arrI.length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOps.setValues(arrI, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 3 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[arrd.length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOps.setValues(arrd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 4 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new double[arri.length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOps.setValues(arri, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 5 ---------------
        arrD = new Double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrD, dest));

        // -------------- sub-case 6 ---------------
        arrI = new Integer[]{1, 344, -9, 1423};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrI, dest));

        // -------------- sub-case 7 ---------------
        arrd = new double[]{1., 1.344, -9.334, 1423.44};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrd, dest));

        // -------------- sub-case 8 ---------------
        arri = new int[]{1, 344, -9, 1423};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arri, dest));
    }


    @Test
    void twoDArrayTestCase() {
        // -------------- sub-case 1 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[arrDD.length*arrDD[0].length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOps.setValues(arrDD, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 2 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new double[arrII.length*arrII[0].length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOps.setValues(arrII, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 3 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[arrdd.length*arrdd[0].length];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        RealDenseSetOps.setValues(arrdd, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 4 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new double[arrii.length*arrii[0].length];
        expDest = new double[]{1, 344, -9, 1423};
        RealDenseSetOps.setValues(arrii, dest);
        assertArrayEquals(expDest, dest);

        // -------------- sub-case 5 ---------------
        arrDD = new Double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrDD, dest));

        // -------------- sub-case 6 ---------------
        arrII = new Integer[][]{{1, 344}, {-9, 1423}};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrII, dest));

        // -------------- sub-case 7 ---------------
        arrdd = new double[][]{{1., 1.344}, {-9.334, 1423.44}};
        dest = new double[25];
        expDest = new double[]{1., 1.344, -9.334, 1423.44};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrdd, dest));

        // -------------- sub-case 8 ---------------
        arrii = new int[][]{{1, 344}, {-9, 1423}};
        dest = new double[25];
        expDest = new double[]{1, 344, -9, 1423};
        assertThrows(IllegalArgumentException.class, ()-> RealDenseSetOps.setValues(arrii, dest));
    }
}
