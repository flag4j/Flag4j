package org.flag4j.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixSwapRowColTests {
    Shape expShape, actShape;
    Complex128[] expData, actData;
    int[] expRowIndices, actRowIndices;
    int[] expColIndices, actColIndices;
    CooCMatrix exp, act;

    @Test
    void swapRowTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(42, 15);
        actData = new Complex128[]{new Complex128(0.83954, 0.27882), new Complex128(0.49378, 0.45173), new Complex128(0.57154, 0.29647), new Complex128(0.53119, 0.32832), new Complex128(0.2354, 0.52295), new Complex128(0.01532, 0.15957)};
        actRowIndices = new int[]{2, 2, 13, 15, 18, 21};
        actColIndices = new int[]{2, 14, 5, 6, 13, 3};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(42, 15);
        expData = new Complex128[]{new Complex128(0.83954, 0.27882), new Complex128(0.49378, 0.45173), new Complex128(0.57154, 0.29647), new Complex128(0.53119, 0.32832), new Complex128(0.2354, 0.52295), new Complex128(0.01532, 0.15957)};
        expRowIndices = new int[]{2, 2, 13, 15, 18, 21};
        expColIndices = new int[]{2, 14, 5, 6, 13, 3};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapRows(0, 5));
        assertEquals(exp, act.swapRows(5, 0));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(13, 15);
        actData = new Complex128[]{new Complex128(0.69961, 0.84185), new Complex128(0.2981, 0.15324), new Complex128(0.38024, 0.52313), new Complex128(0.02953, 0.6424), new Complex128(0.24341, 0.90827), new Complex128(0.34581, 0.51126), new Complex128(0.72608, 0.87854), new Complex128(0.55568, 0.57181), new Complex128(0.37162, 0.5097), new Complex128(0.58337, 0.23366), new Complex128(0.76915, 0.50693), new Complex128(0.4387, 0.26465), new Complex128(0.63875, 0.92151), new Complex128(0.97516, 0.75893), new Complex128(0.5171, 0.82702), new Complex128(0.82809, 0.53152)};
        actRowIndices = new int[]{0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 12, 12, 12};
        actColIndices = new int[]{6, 9, 4, 6, 10, 8, 2, 13, 1, 10, 10, 8, 5, 3, 10, 11};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(13, 15);
        expData = new Complex128[]{new Complex128(0.37162, 0.5097), new Complex128(0.58337, 0.23366), new Complex128(0.38024, 0.52313), new Complex128(0.02953, 0.6424), new Complex128(0.24341, 0.90827), new Complex128(0.34581, 0.51126), new Complex128(0.72608, 0.87854), new Complex128(0.55568, 0.57181), new Complex128(0.69961, 0.84185), new Complex128(0.2981, 0.15324), new Complex128(0.76915, 0.50693), new Complex128(0.4387, 0.26465), new Complex128(0.63875, 0.92151), new Complex128(0.97516, 0.75893), new Complex128(0.5171, 0.82702), new Complex128(0.82809, 0.53152)};
        expRowIndices = new int[]{0, 0, 1, 2, 2, 3, 4, 4, 5, 5, 6, 7, 8, 12, 12, 12};
        expColIndices = new int[]{1, 10, 4, 6, 10, 8, 2, 13, 6, 9, 10, 8, 5, 3, 10, 11};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapRows(0, 5));
        assertEquals(exp, act.swapRows(5, 0));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(13, 15);
        actData = new Complex128[]{new Complex128(0.24099, 0.0519), new Complex128(0.41046, 0.01209), new Complex128(0.44403, 0.26415), new Complex128(0.64789, 0.41217), new Complex128(0.34822, 0.81154), new Complex128(0.73094, 0.6667), new Complex128(0.58335, 0.76636), new Complex128(0.3706, 0.00392), new Complex128(0.805, 0.43727), new Complex128(0.64965, 0.42741), new Complex128(0.63493, 0.04803), new Complex128(0.17587, 0.57598), new Complex128(0.2884, 0.13797), new Complex128(0.74812, 0.06963), new Complex128(0.29151, 0.61113), new Complex128(0.79594, 0.98446), new Complex128(0.32668, 0.48901), new Complex128(0.74668, 0.44182), new Complex128(0.63456, 0.4588), new Complex128(0.2882, 0.3568)};
        actRowIndices = new int[]{1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 12, 12};
        actColIndices = new int[]{5, 1, 9, 14, 13, 14, 7, 12, 14, 8, 10, 8, 1, 2, 3, 12, 0, 1, 7, 8};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(13, 15);
        expData = new Complex128[]{new Complex128(0.24099, 0.0519), new Complex128(0.41046, 0.01209), new Complex128(0.44403, 0.26415), new Complex128(0.64789, 0.41217), new Complex128(0.34822, 0.81154), new Complex128(0.32668, 0.48901), new Complex128(0.74668, 0.44182), new Complex128(0.63456, 0.4588), new Complex128(0.2882, 0.3568), new Complex128(0.58335, 0.76636), new Complex128(0.3706, 0.00392), new Complex128(0.805, 0.43727), new Complex128(0.64965, 0.42741), new Complex128(0.63493, 0.04803), new Complex128(0.17587, 0.57598), new Complex128(0.2884, 0.13797), new Complex128(0.74812, 0.06963), new Complex128(0.29151, 0.61113), new Complex128(0.79594, 0.98446), new Complex128(0.73094, 0.6667)};
        expRowIndices = new int[]{1, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6, 6, 7, 8, 9, 10, 10, 11, 11, 12};
        expColIndices = new int[]{5, 1, 9, 14, 13, 0, 1, 7, 8, 7, 12, 14, 8, 10, 8, 1, 2, 3, 12, 14};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapRows(5, 12));
        assertEquals(exp, act.swapRows(12, 5));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(13, 15);
        actData = new Complex128[]{new Complex128(0.24099, 0.0519), new Complex128(0.41046, 0.01209), new Complex128(0.44403, 0.26415), new Complex128(0.64789, 0.41217), new Complex128(0.34822, 0.81154), new Complex128(0.73094, 0.6667), new Complex128(0.58335, 0.76636), new Complex128(0.3706, 0.00392), new Complex128(0.805, 0.43727), new Complex128(0.64965, 0.42741), new Complex128(0.63493, 0.04803), new Complex128(0.17587, 0.57598), new Complex128(0.2884, 0.13797), new Complex128(0.74812, 0.06963), new Complex128(0.29151, 0.61113), new Complex128(0.79594, 0.98446), new Complex128(0.32668, 0.48901), new Complex128(0.74668, 0.44182), new Complex128(0.63456, 0.4588), new Complex128(0.2882, 0.3568)};
        actRowIndices = new int[]{1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 12, 12};
        actColIndices = new int[]{5, 1, 9, 14, 13, 14, 7, 12, 14, 8, 10, 8, 1, 2, 3, 12, 0, 1, 7, 8};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(13, 15);
        expData = new Complex128[]{new Complex128(0.24099, 0.0519), new Complex128(0.41046, 0.01209), new Complex128(0.44403, 0.26415), new Complex128(0.64789, 0.41217), new Complex128(0.34822, 0.81154), new Complex128(0.73094, 0.6667), new Complex128(0.58335, 0.76636), new Complex128(0.3706, 0.00392), new Complex128(0.805, 0.43727), new Complex128(0.64965, 0.42741), new Complex128(0.63493, 0.04803), new Complex128(0.17587, 0.57598), new Complex128(0.2884, 0.13797), new Complex128(0.74812, 0.06963), new Complex128(0.29151, 0.61113), new Complex128(0.79594, 0.98446), new Complex128(0.32668, 0.48901), new Complex128(0.74668, 0.44182), new Complex128(0.63456, 0.4588), new Complex128(0.2882, 0.3568)};
        expRowIndices = new int[]{1, 2, 3, 3, 4, 5, 6, 6, 6, 7, 8, 9, 10, 10, 11, 11, 12, 12, 12, 12};
        expColIndices = new int[]{5, 1, 9, 14, 13, 14, 7, 12, 14, 8, 10, 8, 1, 2, 3, 12, 0, 1, 7, 8};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapRows(5, 5));
        assertEquals(exp, act.swapRows(12, 12));

        // ------------------ sub-case 4 ------------------
        actShape = new Shape(13, 15);
        act = new CooCMatrix(actShape);

        assertThrows(IndexOutOfBoundsException.class, () -> act.swapRows(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapRows(0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapRows(1, 14));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapRows(13, 1));
    }


    @Test
    void swapColTests() {
        // ------------------ sub-case 1 ------------------
        actShape = new Shape(15, 13);
        actData = new Complex128[]{new Complex128(0.91602, 0.16221), new Complex128(0.96461, 0.28433), new Complex128(0.58055, 0.82438), new Complex128(0.52431, 0.01252), new Complex128(0.29084, 0.55349), new Complex128(0.57376, 0.65649), new Complex128(0.30012, 0.98787), new Complex128(0.04433, 0.47742), new Complex128(0.74451, 0.58319), new Complex128(0.03021, 0.14801), new Complex128(0.02442, 0.49737), new Complex128(0.79838, 0.67637), new Complex128(0.89686, 0.42408), new Complex128(0.50349, 0.86714), new Complex128(0.06367, 0.22256), new Complex128(0.33357, 0.6249), new Complex128(0.69218, 0.04192), new Complex128(0.76516, 0.11117), new Complex128(0.24567, 0.35127), new Complex128(0.56763, 0.58676)};
        actRowIndices = new int[]{0, 0, 1, 3, 4, 5, 5, 6, 6, 6, 8, 9, 10, 10, 11, 12, 13, 14, 14, 14};
        actColIndices = new int[]{3, 12, 4, 4, 8, 0, 12, 1, 4, 5, 7, 9, 0, 11, 0, 7, 4, 2, 6, 11};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(15, 13);
        expData = new Complex128[]{new Complex128(0.91602, 0.16221), new Complex128(0.96461, 0.28433), new Complex128(0.58055, 0.82438), new Complex128(0.52431, 0.01252), new Complex128(0.29084, 0.55349), new Complex128(0.57376, 0.65649), new Complex128(0.30012, 0.98787), new Complex128(0.03021, 0.14801), new Complex128(0.04433, 0.47742), new Complex128(0.74451, 0.58319), new Complex128(0.02442, 0.49737), new Complex128(0.79838, 0.67637), new Complex128(0.89686, 0.42408), new Complex128(0.50349, 0.86714), new Complex128(0.06367, 0.22256), new Complex128(0.33357, 0.6249), new Complex128(0.69218, 0.04192), new Complex128(0.76516, 0.11117), new Complex128(0.24567, 0.35127), new Complex128(0.56763, 0.58676)};
        expRowIndices = new int[]{0, 0, 1, 3, 4, 5, 5, 6, 6, 6, 8, 9, 10, 10, 11, 12, 13, 14, 14, 14};
        expColIndices = new int[]{3, 12, 4, 4, 8, 5, 12, 0, 1, 4, 7, 9, 5, 11, 5, 7, 4, 2, 6, 11};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapCols(0, 5));
        assertEquals(exp, act.swapCols(5, 0));

        // ------------------ sub-case 2 ------------------
        actShape = new Shape(15, 13);
        actData = new Complex128[]{new Complex128(0.11608, 0.61874), new Complex128(0.46118, 0.28252), new Complex128(0.58771, 0.70903), new Complex128(0.77484, 0.16391), new Complex128(0.29803, 0.72315), new Complex128(0.41697, 0.09332), new Complex128(0.16798, 0.65098), new Complex128(0.04629, 0.22586), new Complex128(0.4762, 0.55212), new Complex128(0.88635, 0.57277), new Complex128(0.98888, 0.28049), new Complex128(0.03914, 0.90305), new Complex128(0.44566, 0.92861), new Complex128(0.69347, 0.70765), new Complex128(0.16541, 0.37864), new Complex128(0.02777, 0.03835), new Complex128(0.08825, 0.34296), new Complex128(0.3324, 0.12397), new Complex128(0.79772, 0.86073), new Complex128(0.11217, 0.66587)};
        actRowIndices = new int[]{0, 0, 0, 2, 2, 3, 4, 4, 4, 4, 5, 5, 7, 7, 9, 10, 11, 12, 12, 13};
        actColIndices = new int[]{6, 7, 11, 4, 7, 10, 5, 8, 11, 12, 1, 8, 7, 10, 6, 5, 4, 5, 8, 7};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(15, 13);
        expData = new Complex128[]{new Complex128(0.11608, 0.61874), new Complex128(0.46118, 0.28252), new Complex128(0.58771, 0.70903), new Complex128(0.77484, 0.16391), new Complex128(0.29803, 0.72315), new Complex128(0.41697, 0.09332), new Complex128(0.88635, 0.57277), new Complex128(0.04629, 0.22586), new Complex128(0.4762, 0.55212), new Complex128(0.16798, 0.65098), new Complex128(0.98888, 0.28049), new Complex128(0.03914, 0.90305), new Complex128(0.44566, 0.92861), new Complex128(0.69347, 0.70765), new Complex128(0.16541, 0.37864), new Complex128(0.02777, 0.03835), new Complex128(0.08825, 0.34296), new Complex128(0.79772, 0.86073), new Complex128(0.3324, 0.12397), new Complex128(0.11217, 0.66587)};
        expRowIndices = new int[]{0, 0, 0, 2, 2, 3, 4, 4, 4, 4, 5, 5, 7, 7, 9, 10, 11, 12, 12, 13};
        expColIndices = new int[]{6, 7, 11, 4, 7, 10, 5, 8, 11, 12, 1, 8, 7, 10, 6, 12, 4, 8, 12, 7};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapCols(12, 5));
        assertEquals(exp, act.swapCols(5, 12));


        // ------------------ sub-case 2 ------------------
        actShape = new Shape(15, 13);
        actData = new Complex128[]{new Complex128(0.11608, 0.61874), new Complex128(0.46118, 0.28252), new Complex128(0.58771, 0.70903), new Complex128(0.77484, 0.16391), new Complex128(0.29803, 0.72315), new Complex128(0.41697, 0.09332), new Complex128(0.16798, 0.65098), new Complex128(0.04629, 0.22586), new Complex128(0.4762, 0.55212), new Complex128(0.88635, 0.57277), new Complex128(0.98888, 0.28049), new Complex128(0.03914, 0.90305), new Complex128(0.44566, 0.92861), new Complex128(0.69347, 0.70765), new Complex128(0.16541, 0.37864), new Complex128(0.02777, 0.03835), new Complex128(0.08825, 0.34296), new Complex128(0.3324, 0.12397), new Complex128(0.79772, 0.86073), new Complex128(0.11217, 0.66587)};
        actRowIndices = new int[]{0, 0, 0, 2, 2, 3, 4, 4, 4, 4, 5, 5, 7, 7, 9, 10, 11, 12, 12, 13};
        actColIndices = new int[]{6, 7, 11, 4, 7, 10, 5, 8, 11, 12, 1, 8, 7, 10, 6, 5, 4, 5, 8, 7};
        act = new CooCMatrix(actShape, actData, actRowIndices, actColIndices);

        expShape = new Shape(15, 13);
        expData = new Complex128[]{new Complex128(0.11608, 0.61874), new Complex128(0.46118, 0.28252), new Complex128(0.58771, 0.70903), new Complex128(0.77484, 0.16391), new Complex128(0.29803, 0.72315), new Complex128(0.41697, 0.09332), new Complex128(0.16798, 0.65098), new Complex128(0.04629, 0.22586), new Complex128(0.4762, 0.55212), new Complex128(0.88635, 0.57277), new Complex128(0.98888, 0.28049), new Complex128(0.03914, 0.90305), new Complex128(0.44566, 0.92861), new Complex128(0.69347, 0.70765), new Complex128(0.16541, 0.37864), new Complex128(0.02777, 0.03835), new Complex128(0.08825, 0.34296), new Complex128(0.3324, 0.12397), new Complex128(0.79772, 0.86073), new Complex128(0.11217, 0.66587)};
        expRowIndices = new int[]{0, 0, 0, 2, 2, 3, 4, 4, 4, 4, 5, 5, 7, 7, 9, 10, 11, 12, 12, 13};
        expColIndices = new int[]{6, 7, 11, 4, 7, 10, 5, 8, 11, 12, 1, 8, 7, 10, 6, 5, 4, 5, 8, 7};
        exp = new CooCMatrix(expShape, expData, expRowIndices, expColIndices);

        assertEquals(exp, act.copy().swapCols(5, 5));
        assertEquals(exp, act.swapCols(12, 12));

        // ------------------ sub-case 3 ------------------
        actShape = new Shape(15, 13);
        act = new CooCMatrix(actShape);
        
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapCols(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapCols(0, -1));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapCols(1, 14));
        assertThrows(IndexOutOfBoundsException.class, () -> act.swapCols(13, 1));
    }
}
