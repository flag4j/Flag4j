package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class TensorDotTests {
    static int[] aAxes;
    static int[] bAxes;

    static double[] aEntries;
    static double[] bEntries;
    static double[] expEntries;

    static Shape aShape;
    static Shape bShape;
    static Shape expShape;

    static Tensor A;
    static Tensor B;
    static Tensor exp;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.4415, 235.61, -0.00024, 1.0, -85.1, 1.345,
                0.014, -140.0, 1.5, 51.0, 6.1, -0.00014};
        bEntries = new double[]{
                13.41, -99.23, 14, 0.000245, 1.25, 95.14, 546, 15.6, 1.45656,
                0.009345, 156.21326, 125.6, 144.5, 1545.4, 1.145, 7.4, -9.345,
                0.0, 314.5, 04.155, -0.0, 145.5, 7, 1.4456};

        aShape = new Shape(3, 2, 1, 2);
        bShape = new Shape(4, 3, 2);

        A = new Tensor(aShape, aEntries);
        B = new Tensor(bShape, bEntries);
    }


    @Test
    void multiAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        aAxes = new int[]{0};
        bAxes = new int[]{1};
        expEntries = new double[]{
                -1170.1944849999998, -0.35089449999998124, 897.425634, 210.09214049999997,
                96.83975000000001, 1597.9541000000002, 463.85175, -12373.892167499998,
                3242.1101000000003, -18527.439970475003, 136611.89533320002, 10081.128569025,
                33570.590025000005, 364121.647, 74456.345, 1248.3826500000002,
                7.8177816, 580.37781863, 952.7902378399999, 766.15638683,
                -57.02315, -0.26729600000000003, 42.62452, 10.8541628,
                -1946.5901749999998, -99.27761960000001, 342.0597301436, 14.274116000000001,
                -15.798691700000012, 509.4000000000001, 314.49902, -20365.845202384};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 2 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 2};
        expEntries = new double[]{
                -589.8166663699998, 1663.58202083, 96.57245400000002, 474.70591279999996,
                3142.8324804000003, 136626.16944920004, 34079.99002500001, 54090.499797616};
        expShape = new Shape(1, 2, 4);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 3 ---------------------
        aAxes = new int[]{1};
        bAxes = new int[]{0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 4 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1};

        assertThrows(IllegalArgumentException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 5 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 5 ---------------------
        aAxes = new int[]{0, 1, 2, 3};
        bAxes = new int[]{1, 0, 3, 2};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));
    }


    @Test
    void singleAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new double[]{
                -1170.1944849999998, -0.35089449999998124, 897.425634, 210.09214049999997,
                96.83975000000001, 1597.9541000000002, 463.85175, -12373.892167499998,
                3242.1101000000003, -18527.439970475003, 136611.89533320002, 10081.128569025,
                33570.590025000005, 364121.647, 74456.345, 1248.3826500000002,
                7.8177816, 580.37781863, 952.7902378399999, 766.15638683,
                -57.02315, -0.26729600000000003, 42.62452, 10.8541628,
                -1946.5901749999998, -99.27761960000001, 342.0597301436, 14.274116000000001,
                -15.798691700000012, 509.4000000000001, 314.49902, -20365.845202384};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, 0, 1));


        // --------------------- sub-case 1 ---------------------
        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, 1, 1));
    }


    @Test
    void noAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        expEntries = new double[]{
                3317.870215, -143.00515415, 1130.2016104176003, 24.65903145,
                478.0724428, 3971.2081000000003, 453.35006999999996, 34287.244085556005,
                -1176.3087499999997, 28.712450500000003, 632.1535787, 183.7367405,
                34.49147500000001, 915.6600000000001, 323.915, -12375.950668,
                -1957.93726, 141.28648, 38.04548999999997, 187.31009999999998,
                -172.29450000000003, -1014.3644, 14.903, -20367.77343,
                769.3098249999999, -5060.7418251, 27854.8631461436, 795.6394204999999,
                7376.4858083, 78860.54000000001, 16039.49902, 1099.454797616};
        expShape = new Shape(2, 2, 4, 2);
        exp = new Tensor(expShape, expEntries);
        B = B.reshape(4, 3, 2);

        assertEquals(exp, A.tensorDot(B));


        // --------------------- sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        B = B.reshape(4, 2, 3);
        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B));
    }
}
