package org.flag4j.arrays.dense.tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.arrays.dense.Tensor;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class TensorScalMultDivTests {

    static double[] aEntries;
    static Tensor A;
    static Shape aShape, expShape;

    @BeforeEach
    void setup() {
        aEntries = new double[]{
                1.23, 2.556, -121.5, 15.61, 14.15, -99.23425,
                0.001345, 2.677, 8.14, -0.000194, 1, 234
        };
        aShape = new Shape(2, 3, 2);
        A = new Tensor(aShape, aEntries);
    }


    @Test
    void realScalMultTestCase() {
        double[] expEntries;
        Tensor exp;
        double b = -1.4115;

        // ------------------------ sub-case 1 ------------------------
        expEntries = new double[]{
                1.23*-1.4115, 2.556*-1.4115, -121.5*-1.4115, 15.61*-1.4115, 14.15*-1.4115, -99.23425*-1.4115,
                0.001345*-1.4115, 2.677*-1.4115, 8.14*-1.4115, -0.000194*-1.4115, 1*-1.4115, 234*-1.4115
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.mult(b));
    }


    @Test
    void complexScalMultTestCase() {
        Complex128[] expEntries;
        CTensor exp;
        Complex128 b = new Complex128(0.2425, -0.00295);

        // ------------------------ sub-case 1 ------------------------
        expEntries = new Complex128[]{
                b.mult(1.23), b.mult(2.556), b.mult(-121.5), b.mult(15.61), b.mult(14.15), b.mult(-99.23425),
                b.mult(0.001345), b.mult(2.677), b.mult(8.14), b.mult(-0.000194), b.mult(1), b.mult(234)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.mult(b));
    }


    @Test
    void realScalDivTestCase() {
        double[] expEntries;
        Tensor exp;
        double b = -1.4115;

        // ------------------------ sub-case 1 ------------------------
        expEntries = new double[]{
                1.23*(1.0/-1.4115), 2.556*(1.0/-1.4115), -121.5*(1.0/-1.4115), 15.61*(1.0/-1.4115), 14.15*(1.0/-1.4115), -99.23425*(1.0/-1.4115),
                0.001345*(1.0/-1.4115), 2.677*(1.0/-1.4115), 8.14*(1.0/-1.4115), -0.000194*(1.0/-1.4115), 1*(1.0/-1.4115), 234*(1.0/-1.4115)
        };
        expShape = new Shape(2, 3, 2);
        exp = new Tensor(expShape, expEntries);

        assertEquals(exp, A.div(b));
    }


    @Test
    void complexScalDivTestCase() {
        Complex128[] expEntries;
        CTensor exp;
        Complex128 b = new Complex128(0.2425, -0.00295);
        Complex128 bInv = b.invert();

        // ------------------------ sub-case 1 ------------------------
        expEntries = new Complex128[]{
                new Complex128("5.071414450262457 + 0.06169349537432679i"),
                new Complex128("10.53864661371613 + 0.12820209282664985i"),
                bInv.mult(-121.5),
                bInv.mult(15.61),
                new Complex128("58.341881683913634 + 0.7097259833713205i"),
                new Complex128("-409.15285317964 - 4.977323368577064i"),
                bInv.mult(0.001345),
                new Complex128("11.037541856384227 + 0.13427112773745759i"),
                new Complex128("33.562043597671874 + 0.408280530363431i"),
                bInv.mult(-0.000194),
                bInv.mult(1),
                bInv.mult(234)
        };
        expShape = new Shape(2, 3, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.div(b));
    }
}
