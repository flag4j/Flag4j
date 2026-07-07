package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class CooCMatrixMultTransposeTests {


    @Test
    void complexSparseMultTransposeTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        Complex128[] aEntries;
        CooCMatrix a;

        Shape bShape;
        int[] bRowIndices;
        int[] bColIndices;
        Complex128[] bEntries;
        CooCMatrix b;

        Complex128[][] expEntries;
        CMatrix exp;

        // ---------------------  sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.13333+0.54407i"), new Complex128("0.28186+0.80017i"), new Complex128("0.27979+0.90149i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{2, 4, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(6, 5);
        bEntries = new Complex128[]{new Complex128("0.7783+0.53458i"), new Complex128("0.90399+0.64094i"), new Complex128("0.51196+0.51884i"), new Complex128("0.46424+0.32518i"), new Complex128("0.69994+0.96631i"), new Complex128("0.89585+0.321i"), new Complex128("0.29658+0.1626i"), new Complex128("0.70076+0.3312i"), new Complex128("0.35692+0.11799i"), new Complex128("0.83837+0.80204i")};
        bRowIndices = new int[]{0, 1, 2, 3, 4, 4, 4, 4, 5, 5};
        bColIndices = new int[]{4, 0, 2, 1, 0, 1, 3, 4, 0, 3};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);
        expEntries = new Complex128[][]{
                {new Complex128(0.6471265166000001, 0.4720955922000001), new Complex128(0.0, 0.0), new Complex128(0.35054490559999996, 0.20936514), new Complex128(0.0, 0.0), new Complex128(0.4625325176, 0.46737509720000014), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.8307283627, 0.6356093425), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(1.0669550145, 0.3606250357000001), new Complex128(0.2062294519, 0.28874738870000005)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)}
        };

        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  sub-case 2 ---------------------
        aShape = new Shape(11, 23);
        aEntries = new Complex128[]{new Complex128("0.1091+0.67734i"), new Complex128("0.86221+0.18731i"), new Complex128("0.86577+0.3935i"), new Complex128("0.79207+0.16091i"), new Complex128("0.15273+0.23584i"), new Complex128("0.40546+0.55037i"), new Complex128("0.06335+0.55614i"), new Complex128("0.45687+0.32064i"), new Complex128("0.84184+0.56079i"), new Complex128("0.53839+0.56551i"), new Complex128("0.82059+0.48871i"), new Complex128("0.19813+0.82157i"), new Complex128("0.23558+0.75971i"), new Complex128("0.89675+0.70556i"), new Complex128("0.30268+0.21512i")};
        aRowIndices = new int[]{0, 0, 0, 1, 1, 2, 2, 2, 4, 5, 6, 7, 8, 10, 10};
        aColIndices = new int[]{2, 7, 8, 0, 10, 3, 14, 17, 2, 22, 13, 14, 0, 4, 20};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(11, 23);
        bEntries = new Complex128[]{new Complex128("0.09315+0.56922i"), new Complex128("0.11107+0.98862i"), new Complex128("0.32474+0.35843i"), new Complex128("0.79492+0.42901i"), new Complex128("0.82111+0.21961i"), new Complex128("0.58618+0.95779i"), new Complex128("0.05473+0.03025i"), new Complex128("0.97621+0.86671i"), new Complex128("0.80408+0.59324i"), new Complex128("0.03767+0.67623i"), new Complex128("0.61841+0.27712i"), new Complex128("0.88397+0.64611i"), new Complex128("0.56549+0.54455i"), new Complex128("0.09233+0.22337i"), new Complex128("0.41633+0.25273i"), new Complex128("0.56249+0.55354i"), new Complex128("0.62517+0.56234i"), new Complex128("0.69762+0.90077i"), new Complex128("0.89853+0.70042i"), new Complex128("0.79632+0.18535i"), new Complex128("0.22337+0.65651i"), new Complex128("0.98515+0.5477i"), new Complex128("0.23836+0.22636i"), new Complex128("0.37038+0.87641i"), new Complex128("0.49897+0.72714i"), new Complex128("0.38397+0.13316i"), new Complex128("0.05107+0.31637i"), new Complex128("0.26104+0.52205i"), new Complex128("0.30751+0.82356i"), new Complex128("0.43528+0.9727i"), new Complex128("0.29987+0.92454i"), new Complex128("0.42656+0.15536i"), new Complex128("0.76042+0.01935i"), new Complex128("0.46552+0.85124i"), new Complex128("0.52773+0.92759i"), new Complex128("0.11239+0.98725i"), new Complex128("0.3875+0.27288i"), new Complex128("0.52375+0.34944i"), new Complex128("0.49127+0.69019i"), new Complex128("0.45659+0.25621i"), new Complex128("0.18231+0.12218i"), new Complex128("0.22033+0.51241i"), new Complex128("0.83535+0.08698i"), new Complex128("0.12028+0.55878i"), new Complex128("0.06383+0.67314i"), new Complex128("0.70707+0.31567i"), new Complex128("0.70939+0.15072i"), new Complex128("0.16456+0.52904i"), new Complex128("0.98872+0.05189i"), new Complex128("0.71359+0.01168i"), new Complex128("0.88549+0.86375i"), new Complex128("0.09344+0.08053i"), new Complex128("0.72458+0.6146i"), new Complex128("0.69591+0.15405i"), new Complex128("0.08155+0.51061i"), new Complex128("0.09773+0.01328i"), new Complex128("0.51441+0.35341i"), new Complex128("0.88777+0.58754i"), new Complex128("0.35335+0.27309i"), new Complex128("0.40736+0.04797i"), new Complex128("0.21963+0.78092i"), new Complex128("0.5233+0.52295i"), new Complex128("0.4413+0.23607i"), new Complex128("0.41783+0.78562i"), new Complex128("0.69526+0.53609i"), new Complex128("0.81308+0.77482i"), new Complex128("0.08673+0.26201i"), new Complex128("0.6984+0.37514i"), new Complex128("0.17061+0.67571i"), new Complex128("0.28942+0.7071i"), new Complex128("0.83855+0.65238i"), new Complex128("0.49973+0.6632i"), new Complex128("0.45344+0.4296i"), new Complex128("0.54924+0.01977i"), new Complex128("0.86923+0.24688i"), new Complex128("0.82967+0.21481i"), new Complex128("0.16618+0.51186i"), new Complex128("0.03249+0.94928i"), new Complex128("0.11377+0.88986i"), new Complex128("0.47167+0.20325i"), new Complex128("0.34062+0.91378i"), new Complex128("0.60451+0.33083i"), new Complex128("0.09217+0.30043i"), new Complex128("0.24986+0.79771i"), new Complex128("0.73108+0.74435i"), new Complex128("0.01894+0.89016i"), new Complex128("0.21863+0.62992i"), new Complex128("0.8874+0.15921i"), new Complex128("0.34217+0.47171i")};
        bRowIndices = new int[]{0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10};
        bColIndices = new int[]{4, 5, 11, 21, 22, 3, 4, 8, 9, 11, 12, 14, 19, 20, 3, 5, 6, 8, 9, 12, 14, 16, 17, 19, 22, 3, 7, 18, 22, 0, 4, 5, 6, 10, 14, 15, 17, 21, 22, 4, 5, 10, 11, 12, 16, 17, 19, 20, 2, 6, 7, 9, 12, 14, 15, 17, 19, 2, 5, 7, 8, 9, 10, 12, 16, 19, 20, 22, 2, 10, 15, 18, 20, 2, 3, 5, 6, 14, 16, 17, 18, 21, 6, 8, 15, 16, 19, 21, 22};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128(0.0, 0.0), new Complex128(1.1862237167, -0.3662328817), new Complex128(0.9584314624000001, -0.5053461729), new Complex128(0.1032923294, -0.263211456), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(1.06828387, 0.08516565020000022), new Complex128(1.352476262, -0.017508797699999912), new Complex128(0.4762989624000001, 0.041841016400000014), new Complex128(0.07331309579999999, 0.3698653146), new Complex128(0.5302201772, -0.5923134767)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.7731446978000001, -0.7206272326000001), new Complex128(0.1544977753, -0.026297752100000002), new Complex128(0.0, 0.0), new Complex128(0.12307449780000002, 0.0680212209), new Complex128(0.21096558059999998, -0.039738570200000004), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(1.18013854, 0.3849503605), new Complex128(0.8686417363999999, 0.18230937679999998), new Complex128(0.22897174539999998, 0.1573345153), new Complex128(0.8138349663, 0.2343062501), new Complex128(0.4242554997, 0.08249477189999996), new Complex128(0.1786672698, 0.4025332335), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(1.2989661149999998, 0.3946076922), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.8614434379, 0.5107812112), new Complex128(1.0768468534, 0.0032378646999999816), new Complex128(0.5225577333, -0.47316332450000004), new Complex128(0.4734590199, 0.2913651228), new Complex128(0.0, 0.0)},
                {new Complex128(0.566269064, 0.3461100882), new Complex128(0.0, 0.0), new Complex128(0.6798453997, -0.10931237990000003), new Complex128(0.6312917244999999, -0.26949648830000006), new Complex128(0.6548042022, -0.09377329640000004), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.5881569974, 0.19298055939999997), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.4509776284, -0.06046339020000005)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.7059655687999999, 0.5982294586000001), new Complex128(0.5836252188000001, 0.053439764599999995), new Complex128(0.0, 0.0), new Complex128(0.8666392612, 0.24978372940000002), new Complex128(0.0, 0.0), new Complex128(0.2644435068, 0.5412168522), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.7863372133000001, -0.1613880371), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.8415131794, 0.10153790280000002), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)},
                {new Complex128(0.48515112569999996, -0.44472512100000006), new Complex128(0.1464201163, -0.03625899070000002), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0), new Complex128(0.9212268649, -0.6175049678), new Complex128(0.7538347157, -0.03233435709999999), new Complex128(0.0, 0.0), new Complex128(0.0826150276, -0.06064782920000001), new Complex128(0.2296627712, -0.03248731519999999), new Complex128(0.0, 0.0), new Complex128(0.0, 0.0)}
        };
        exp = new CMatrix(expEntries);

        CMatrix act = a.multTranspose(b);
        assertTrue(exp.allClose(act));

        // ---------------------  sub-case 3 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.84627+0.52397i"), new Complex128("0.65009+0.84477i"), new Complex128("0.66607+0.81437i"), new Complex128("0.1052+0.24359i")};
        aRowIndices = new int[]{0, 2, 3, 3};
        aColIndices = new int[]{0, 1, 0, 1};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(1, 3);
        bEntries = new Complex128[]{new Complex128("0.78083+0.73775i")};
        bRowIndices = new int[]{0};
        bColIndices = new int[]{2};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("0.0")},
                {new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  sub-case 4 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.02414+0.82288i"), new Complex128("0.94529+0.19391i"), new Complex128("0.1772+0.50064i"), new Complex128("0.09017+0.45946i")};
        aRowIndices = new int[]{2, 3, 3, 3};
        aColIndices = new int[]{0, 0, 1, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(2, 3);
        bEntries = new Complex128[]{new Complex128("0.65184+0.97107i"), new Complex128("0.70615+0.94007i")};
        bRowIndices = new int[]{1, 1};
        bColIndices = new int[]{0, 1};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(0.0), new Complex128(0.0)},
                {new Complex128(0.0), new Complex128(0.8148094991999999, 0.5129444693999999)},
                {new Complex128(0.0), new Complex128(1.4002444421, -0.6045979338999999)},
                {new Complex128(0.0), new Complex128(0.0)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, a.multTranspose(b));

        // ---------------------  sub-case 5 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.43846+0.30964i"), new Complex128("0.76753+0.33258i"), new Complex128("0.92751+0.69248i"), new Complex128("0.18901+0.72498i")};
        aRowIndices = new int[]{0, 1, 1, 4};
        aColIndices = new int[]{1, 1, 2, 2};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 2);
        bEntries = new Complex128[]{new Complex128("0.18351+0.73322i"), new Complex128("0.59183+0.61502i"), new Complex128("0.1071+0.90921i"), new Complex128("0.34189+0.85244i")};
        bRowIndices = new int[]{0, 2, 3, 4};
        bColIndices = new int[]{1, 1, 1, 0};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final0a = a;
        CooCMatrix final0b = b;
        assertThrows(Exception.class, ()->final0a.multTranspose(final0b));

        // ---------------------  sub-case 6 ---------------------
        aShape = new Shape(5, 3);
        aEntries = new Complex128[]{new Complex128("0.65534+0.28091i"), new Complex128("0.29233+0.20161i"), new Complex128("0.10737+0.28494i"), new Complex128("0.24774+0.02259i")};
        aRowIndices = new int[]{1, 3, 3, 4};
        aColIndices = new int[]{0, 0, 1, 0};
        a = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128("0.27254+0.44057i"), new Complex128("0.49021+0.16341i"), new Complex128("0.14641+0.73422i"), new Complex128("0.90959+0.99102i"), new Complex128("0.46919+0.61656i"), new Complex128("0.79621+0.30459i"), new Complex128("0.36651+0.7489i"), new Complex128("0.01621+0.51064i"), new Complex128("0.3416+0.99506i")};
        bRowIndices = new int[]{0, 0, 1, 2, 2, 2, 2, 3, 3};
        bColIndices = new int[]{3, 4, 0, 1, 2, 3, 4, 2, 4};
        b = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix final1a = a;
        CooCMatrix final1b = b;
        assertThrows(Exception.class, ()->final1a.multTranspose(final1b));
    }
}
