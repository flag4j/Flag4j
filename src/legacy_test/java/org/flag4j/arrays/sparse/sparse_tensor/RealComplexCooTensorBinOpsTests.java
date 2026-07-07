package org.flag4j.arrays.sparse.sparse_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.arrays.sparse.CooTensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealComplexCooTensorBinOpsTests {

    CooTensor A;
    Shape aShape;
    double[] aEntries;
    int[][] aIndices;

    CooCTensor B;
    Shape bShape;
    Complex128[] bEntries;
    int[][] bIndices;

    CooCTensor exp;
    Shape expShape;
    Complex128[] expEntries;
    int[][] expIndices;

    @Test
    void addTests() {
        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(3, 16);
        aEntries = new double[]{1.7463475926857943, -1.130295792343086, -1.4629436086045846, -0.3100760165046801, -0.09665657433974414, 0.13476509808719928, -0.0638077132075331};
        aIndices = new int[][]{
                {0, 2},
                {0, 13},
                {0, 15},
                {1, 5},
                {1, 13},
                {2, 1},
                {2, 3}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 16);
        bEntries = new Complex128[]{new Complex128(0.318, 0.9927), new Complex128(0.3394, 0.6087), new Complex128(0.2087, 0.4036), new Complex128(0.8053, 0.9675), new Complex128(0.474, 0.0026), new Complex128(0.1479, 0.0052), new Complex128(0.0073, 0.7328)};
        bIndices = new int[][]{
                {0, 9},
                {0, 12},
                {1, 3},
                {1, 4},
                {2, 6},
                {2, 9},
                {2, 11}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 16);
        expEntries = new Complex128[]{new Complex128(1.7463475926857943, 0.0), new Complex128(0.318, 0.9927), new Complex128(0.3394, 0.6087), new Complex128(-1.130295792343086, 0.0), new Complex128(-1.4629436086045846, 0.0), new Complex128(0.2087, 0.4036), new Complex128(0.8053, 0.9675), new Complex128(-0.3100760165046801, 0.0), new Complex128(-0.09665657433974414, 0.0), new Complex128(0.13476509808719928, 0.0), new Complex128(-0.0638077132075331, 0.0), new Complex128(0.474, 0.0026), new Complex128(0.1479, 0.0052), new Complex128(0.0073, 0.7328)};
        expIndices = new int[][]{
                {0, 2},
                {0, 9},
                {0, 12},
                {0, 13},
                {0, 15},
                {1, 3},
                {1, 4},
                {1, 5},
                {1, 13},
                {2, 1},
                {2, 3},
                {2, 6},
                {2, 9},
                {2, 11}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.add(B));

        // -------------------------- sub-case 1 --------------------------
        aShape = new Shape(5, 3, 2, 1, 8);
        aEntries = new double[]{1.605979015236944, -1.2944465840036907, 0.18895750482591192, -0.7307330521229386, -0.4500047202309869, -0.2760206404103061, -0.05495875438029267, 0.4398947793476673, 1.485028116781792, 2.0853070263385405, -1.295509717819037, -0.19473190000656226};
        aIndices = new int[][]{
                {0, 0, 0, 0, 2},
                {1, 0, 0, 0, 6},
                {1, 1, 1, 0, 1},
                {2, 1, 1, 0, 5},
                {2, 2, 0, 0, 2},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 1},
                {3, 0, 0, 0, 6},
                {3, 1, 0, 0, 6},
                {3, 1, 1, 0, 0},
                {3, 2, 0, 0, 0},
                {4, 2, 0, 0, 0}};
        A = new CooTensor(aShape, aEntries, aIndices);

        bShape = new Shape(5, 3, 2, 1, 8);
        bEntries = new Complex128[]{new Complex128(0.2165, 0.1597), new Complex128(0.2, 0.8495), new Complex128(0.6178, 0.9117), new Complex128(0.7567, 0.522), new Complex128(0.5328, 0.1192), new Complex128(0.0001, 0.3608), new Complex128(0.8431, 0.8437), new Complex128(0.9877, 0.5206), new Complex128(0.2649, 0.8721), new Complex128(0.1699, 0.1777), new Complex128(0.1255, 0.0734), new Complex128(0.0357, 0.9614)};
        bIndices = new int[][]{
                {0, 1, 0, 0, 0},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 1},
                {1, 2, 1, 0, 5},
                {2, 1, 0, 0, 5},
                {3, 1, 0, 0, 5},
                {3, 1, 0, 0, 7},
                {3, 1, 1, 0, 3},
                {3, 2, 1, 0, 7},
                {4, 1, 0, 0, 7},
                {4, 2, 1, 0, 2},
                {4, 2, 1, 0, 3}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(5, 3, 2, 1, 8);
        expEntries = new Complex128[]{new Complex128(1.605979015236944, 0.0), new Complex128(0.2165, 0.1597), new Complex128(0.2, 0.8495), new Complex128(0.6178, 0.9117), new Complex128(-1.2944465840036907, 0.0), new Complex128(0.18895750482591192, 0.0), new Complex128(0.7567, 0.522), new Complex128(0.5328, 0.1192), new Complex128(-0.7307330521229386, 0.0), new Complex128(-0.4500047202309869, 0.0), new Complex128(-0.2760206404103061, 0.0), new Complex128(-0.05495875438029267, 0.0), new Complex128(0.4398947793476673, 0.0), new Complex128(0.0001, 0.3608), new Complex128(1.485028116781792, 0.0), new Complex128(0.8431, 0.8437), new Complex128(2.0853070263385405, 0.0), new Complex128(0.9877, 0.5206), new Complex128(-1.295509717819037, 0.0), new Complex128(0.2649, 0.8721), new Complex128(0.1699, 0.1777), new Complex128(-0.19473190000656226, 0.0), new Complex128(0.1255, 0.0734), new Complex128(0.0357, 0.9614)};
        expIndices = new int[][]{
                {0, 0, 0, 0, 2},
                {0, 1, 0, 0, 0},
                {0, 1, 0, 0, 2},
                {0, 1, 1, 0, 1},
                {1, 0, 0, 0, 6},
                {1, 1, 1, 0, 1},
                {1, 2, 1, 0, 5},
                {2, 1, 0, 0, 5},
                {2, 1, 1, 0, 5},
                {2, 2, 0, 0, 2},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 1},
                {3, 0, 0, 0, 6},
                {3, 1, 0, 0, 5},
                {3, 1, 0, 0, 6},
                {3, 1, 0, 0, 7},
                {3, 1, 1, 0, 0},
                {3, 1, 1, 0, 3},
                {3, 2, 0, 0, 0},
                {3, 2, 1, 0, 7},
                {4, 1, 0, 0, 7},
                {4, 2, 0, 0, 0},
                {4, 2, 1, 0, 2},
                {4, 2, 1, 0, 3}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.add(B));
    }
}
