package org.flag4j.arrays.sparse.complex_coo_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.numbers.Complex128;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CooCMatrixElemMultTests {

    Shape aShape, bShape, expShape;
    Complex128[] aEntries;
    CooCMatrix A;
    int[] aRowIndices, aColIndices, bRowIndices, bColIndices, expRowIndices, expColIndices;

    @Test
    void complexSparseRealSparseElemMultTest() {
        double[] bEntries;
        CooMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.5270700221534264+0.11452189627998122i"), new Complex128("0.6325521342685327+0.2858604816625596i"), new Complex128("0.8907136951313099+0.8049207201434311i"), new Complex128("0.9859077804744598+0.3405085838939409i"), new Complex128("0.015153343350000181+0.9558802534337969i")};
        aRowIndices = new int[]{2, 2, 3, 4, 4};
        aColIndices = new int[]{1, 2, 1, 1, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new double[]{0.15609311001425052, 0.2813970726614028, 0.7980488598859438, 0.8292881848250093, 0.2691308899027348};
        bRowIndices = new int[]{1, 2, 2, 3, 4};
        bColIndices = new int[]{4, 0, 3, 1, 0};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.7386583434342208+0.6675112429357852i")};
        expRowIndices = new int[]{3};
        expColIndices = new int[]{1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.31313009812352754+0.9652779017016226i"), new Complex128("0.7973755259034561+0.38991088473694024i"), new Complex128("0.5897612620167405+0.19740472101140905i")};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{3, 2, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new double[]{0.015883143477422035, 0.14139653279196096, 0.7546508674564075};
        bRowIndices = new int[]{0, 1, 2};
        bColIndices = new int[]{4, 2, 3};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.32671443199591466+0.013124290126396043i"), new Complex128("0.5479911723347548+0.186110110015276i"), new Complex128("0.7889900145302261+0.24350374650553708i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 2, 1};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 3);
        bEntries = new double[]{0.2840502547904079, 0.9261546753361493, 0.4363471822672774};
        bRowIndices = new int[]{0, 1, 3};
        bColIndices = new int[]{1, 2, 2};
        B = new CooMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooMatrix finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void complexSparseComplexSparseElemMultTest() {
        Complex128[] bEntries;
        CooCMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.4917044142106619+0.08994193562416375i"), new Complex128("0.19108469417205942+0.6222149542769521i"), new Complex128("0.34074684806929945+0.19566342591325037i"), new Complex128("0.07850107184359123+0.9840242775451442i"), new Complex128("0.4367941908315287+0.7895259084044752i")};
        aRowIndices = new int[]{0, 0, 1, 3, 4};
        aColIndices = new int[]{2, 3, 0, 1, 3};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128("0.4036274179740017+0.7682494794218789i"), new Complex128("0.08465267833118417+0.3828690235304937i"), new Complex128("0.28632691884342554+0.5791229723876221i"), new Complex128("0.9624631916299103+0.5759396358004059i"), new Complex128("0.7754293409607161+0.5942839678524314i")};
        bRowIndices = new int[]{0, 0, 3, 3, 4};
        bColIndices = new int[]{0, 4, 2, 3, 2};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{};
        expRowIndices = new int[]{};
        expColIndices = new int[]{};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.0124565908959402+0.32787058729871454i"), new Complex128("0.3654808891596326+0.23609197778112512i"), new Complex128("0.4658578331275002+0.5456694519065678i")};
        aRowIndices = new int[]{0, 0, 1};
        aColIndices = new int[]{1, 3, 1};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(3, 5);
        bEntries = new Complex128[]{new Complex128("0.638685378683674+0.20084202060653955i"), new Complex128("0.10939208977055415+0.4514905425022917i"), new Complex128("0.6184259742411935+0.7815228507226981i")};
        bRowIndices = new int[]{0, 0, 2};
        bColIndices = new int[]{2, 3, 2};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("-0.06661257689246669+0.190837759748443i")};
        expRowIndices = new int[]{0};
        expColIndices = new int[]{3};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.6769120049521917+0.13251796745470068i"), new Complex128("0.8979490786998655+0.07902727327331793i"), new Complex128("0.9299628996565666+0.5279619879332736i")};
        aRowIndices = new int[]{2, 2, 2};
        aColIndices = new int[]{0, 2, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bShape = new Shape(5, 3);
        bEntries = new Complex128[]{new Complex128("0.5979726709964365+0.17017603699686668i"), new Complex128("0.6646215930127696+0.28348429931779084i"), new Complex128("0.9172673649429806+0.8912791442051502i")};
        bRowIndices = new int[]{0, 3, 4};
        bColIndices = new int[]{0, 0, 1};
        B = new CooCMatrix(bShape, bEntries, bRowIndices, bColIndices);

        CooCMatrix finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void complexSparseRealDenseElemMultTest() {
        double[][] bEntries;
        Matrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.6776687963460167+0.8256632034174329i"), new Complex128("0.3740334201622014+0.15531677143503697i"), new Complex128("0.3924131906217284+0.4524476994268528i"), new Complex128("0.5988791547450321+0.8688574898387552i"), new Complex128("0.23628053258249992+0.4954374703387735i")};
        aRowIndices = new int[]{2, 2, 2, 3, 4};
        aColIndices = new int[]{1, 2, 3, 4, 1};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.31599, 0.26914, 0.29983, 0.29646, 0.79026},
                {0.31377, 0.75426, 0.91047, 0.32213, 0.57627},
                {0.76759, 0.1726, 0.37245, 0.19359, 0.99706},
                {0.75155, 0.27489, 0.50946, 0.68393, 0.6838},
                {0.05548, 0.11324, 0.58205, 0.49111, 0.81538}};
        B = new Matrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("0.11696563424932249+0.14250946890984892i"), new Complex128("0.13930874733941193+0.05784773152097952i"), new Complex128("0.0759672695724604+0.08758935013204444i"), new Complex128("0.4095135660146529+0.5941247515517408i"), new Complex128("0.02675640750964229+0.05610333914116271i")};
        expRowIndices = new int[]{2, 2, 2, 3, 4};
        expColIndices = new int[]{1, 2, 3, 4, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.050075456404789054+0.9369595233045618i"), new Complex128("0.5958212904221909+0.6139071108801012i"), new Complex128("0.18068259326815117+0.8592729833322025i")};
        aRowIndices = new int[]{1, 2, 2};
        aColIndices = new int[]{4, 0, 1};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.37556, 0.13522, 0.92971, 0.71404, 0.25637},
                {0.8007, 0.72031, 0.51066, 0.35033, 0.33407},
                {0.69917, 0.74251, 0.88645, 0.77861, 0.59576}};
        B = new Matrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.016728707721147876+0.3130100679503549i"), new Complex128("0.4165803716244832+0.42922543471404034i"), new Complex128("0.13415863232753492+0.6380187828539937i")};
        expRowIndices = new int[]{1, 2, 2};
        expColIndices = new int[]{4, 0, 1};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.00597408087347262+0.9934806190390506i"), new Complex128("0.9256133985790327+0.903916718269249i"), new Complex128("0.49261639135994273+0.2605553183641669i")};
        aRowIndices = new int[]{0, 0, 2};
        aColIndices = new int[]{2, 3, 2};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new double[][]{
                {0.92082, 0.17446, 0.90881},
                {0.97737, 0.59805, 0.83101},
                {0.21363, 0.82021, 0.28655},
                {0.84183, 0.68944, 0.76812},
                {0.10191, 0.26459, 0.49511}};
        B = new Matrix(bEntries);

        Matrix finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }


    @Test
    void complexSparseComplexDenseElemMultTest() {
        Complex128[][] bEntries;
        CMatrix B;
        Complex128[] expEntries;
        CooCMatrix exp;

        // ------------------- sub-case 1 -------------------
        aShape = new Shape(5, 5);
        aEntries = new Complex128[]{new Complex128("0.6064368081976178+0.4696144681076404i"), new Complex128("0.36085497779308606+0.09636990152488645i"), new Complex128("0.1727737596859793+0.7977188983111609i"), new Complex128("0.5940313486520278+0.8074021642942133i"), new Complex128("0.7803451642129308+0.4093163488573416i")};
        aRowIndices = new int[]{0, 1, 3, 3, 4};
        aColIndices = new int[]{0, 1, 0, 2, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.13975+0.46536i"), new Complex128("0.03019+0.7464i"), new Complex128("0.75641+0.79137i"), new Complex128("0.61375+0.17819i"), new Complex128("0.04337+0.12826i")},
                {new Complex128("0.7321+0.07234i"), new Complex128("0.98302+0.55616i"), new Complex128("0.25079+0.23251i"), new Complex128("0.35445+0.40171i"), new Complex128("0.65138+0.81683i")},
                {new Complex128("0.55021+0.12483i"), new Complex128("0.19732+0.55109i"), new Complex128("0.27109+0.93408i"), new Complex128("0.81691+0.69568i"), new Complex128("0.18641+0.16012i")},
                {new Complex128("0.4177+0.14159i"), new Complex128("0.26966+0.112i"), new Complex128("0.20453+0.53959i"), new Complex128("0.13089+0.65059i"), new Complex128("0.77987+0.76195i")},
                {new Complex128("0.91098+0.75196i"), new Complex128("0.94837+0.12534i"), new Complex128("0.27312+0.55549i"), new Complex128("0.52679+0.46955i"), new Complex128("0.68254+0.85077i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(5, 5);
        expEntries = new Complex128[]{new Complex128("-0.13379024493295444+0.34784005498088616i"), new Complex128("0.3011305758380786+0.2954266450463966i"), new Complex128("-0.04078141939104371+0.3576702204585097i"), new Complex128("-0.31416890209171533+0.4856713400822431i"), new Complex128("0.18438271826453323+0.9432690361065251i")};
        expRowIndices = new int[]{0, 1, 3, 3, 4};
        expColIndices = new int[]{0, 1, 0, 2, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 2 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.6085238714460345+0.8879677482952991i"), new Complex128("0.07118152034365255+0.7013737869435294i"), new Complex128("0.1628457620477497+0.1921549100035057i")};
        aRowIndices = new int[]{0, 1, 1};
        aColIndices = new int[]{4, 1, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.27477+0.16189i"), new Complex128("0.71105+0.53336i"), new Complex128("0.10729+0.75878i"), new Complex128("0.5911+0.20178i"), new Complex128("0.97727+0.15181i")},
                {new Complex128("0.56431+0.79066i"), new Complex128("0.95967+0.44147i"), new Complex128("0.03733+0.78274i"), new Complex128("0.57901+0.7797i"), new Complex128("0.87872+0.74874i")},
                {new Complex128("0.65326+0.79008i"), new Complex128("0.72003+0.58285i"), new Complex128("0.36831+0.63813i"), new Complex128("0.41247+0.49563i"), new Complex128("0.85605+0.76946i")}};
        B = new CMatrix(bEntries);

        expShape = new Shape(3, 5);
        expEntries = new Complex128[]{new Complex128("0.4598897399793568+0.9601642503007695i"), new Complex128("-0.24132471609376682+0.7045118879022092i"), new Complex128("-0.0007782392894262746+0.29077949839391265i")};
        expRowIndices = new int[]{0, 1, 1};
        expColIndices = new int[]{4, 1, 4};
        exp = new CooCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------- sub-case 3 -------------------
        aShape = new Shape(3, 5);
        aEntries = new Complex128[]{new Complex128("0.6874431663899189+0.9440122836405981i"), new Complex128("0.46836138768035107+0.5014775165050014i"), new Complex128("0.08460981750359708+0.07825339704496193i")};
        aRowIndices = new int[]{0, 2, 2};
        aColIndices = new int[]{2, 0, 4};
        A = new CooCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.71541+0.87577i"), new Complex128("0.43145+0.51159i"), new Complex128("0.51428+0.82404i")},
                {new Complex128("0.61755+0.92568i"), new Complex128("0.06407+0.81918i"), new Complex128("0.30904+0.65306i")},
                {new Complex128("0.67732+0.66438i"), new Complex128("0.09503+0.15445i"), new Complex128("0.56095+0.55138i")},
                {new Complex128("0.43417+0.89241i"), new Complex128("0.82971+0.0187i"), new Complex128("0.77379+0.28833i")},
                {new Complex128("0.09857+0.92289i"), new Complex128("0.48516+0.6224i"), new Complex128("0.28046+0.73755i")}};
        B = new CMatrix(bEntries);

        CMatrix finalB = B;
        assertThrows(Exception.class, ()->A.elemMult(finalB));
    }
}
