package org.flag4j.arrays.sparse.csr_matrix;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.linalg.ops.dense_sparse.csr.real_field_ops.RealFieldDenseCsrMatMult;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class RealComplexCsrDenseMatMultTests {

    CsrMatrix A;
    Shape aShape;
    double[] aEntries;
    int[] aRowPointers;
    int[] aColIndices;

    CMatrix B;
    Complex128[][] bEntries;

    CMatrix exp;
    Complex128[][] expEntries;

    @Test
    void standardTests() {
        // ------------------------ sub-case 1  ------------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.40775, 0.00066, 0.90251, 0.83958, 0.50831, 0.33647, 0.04011, 0.62236};
        aRowPointers = new int[]{0, 3, 5, 5, 7, 8};
        aColIndices = new int[]{1, 3, 4, 0, 4, 1, 4, 3};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.48204+0.59198i"), new Complex128("0.66808+0.35812i"), new Complex128("0.28052+0.3799i"), new Complex128("0.59171+0.66545i"), new Complex128("0.57884+0.73151i")},
                {new Complex128("0.77302+0.60005i"), new Complex128("0.61617+0.36969i"), new Complex128("0.56317+0.19616i"), new Complex128("0.61275+0.4591i"), new Complex128("0.75275+0.95003i")},
                {new Complex128("0.46551+0.54915i"), new Complex128("0.95026+0.3433i"), new Complex128("0.93448+0.43158i"), new Complex128("0.8449+0.8074i"), new Complex128("0.07347+0.0873i")},
                {new Complex128("0.66999+0.13998i"), new Complex128("0.08384+0.55878i"), new Complex128("0.08645+0.38561i"), new Complex128("0.89806+0.23426i"), new Complex128("0.0235+0.29708i")},
                {new Complex128("0.05227+0.40564i"), new Complex128("0.87567+0.41385i"), new Complex128("0.45123+0.28923i"), new Complex128("0.34363+0.6419i"), new Complex128("0.92574+0.99279i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("0.3628152961+0.6108569307i"), new Complex128("1.0415995836+0.5246136558000001i"), new Complex128("0.6369292118000001+0.3412717099i"), new Complex128("0.5605710434+0.7666738056000001i"), new Complex128("1.1424389299+1.2835737082i")},
                {new Complex128("0.4312805069+0.7032054368i"), new Complex128("1.0060184241+0.5110344831i"), new Complex128("0.46488370290000003+0.4659749433i"), new Complex128("0.6714584470999999+0.8849827000000001i"), new Complex128("0.9565453866000001+1.1188062507i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.2621945891+0.2181690439i"), new Complex128("0.2424458436+0.1409891178i"), new Complex128("0.20758864519999998+0.07760297050000001i"), new Complex128("0.2199549918+0.180219986i"), new Complex128("0.2904092239+0.35947740100000003i")},
                {new Complex128("0.4169749764+0.0871179528i"), new Complex128("0.0521786624+0.3477623208i"), new Complex128("0.053803022+0.23998823960000001i"), new Complex128("0.5589166216+0.1457940536i"), new Complex128("0.01462546+0.18489070880000003i")}};
        exp = new CMatrix(expEntries);

        Assertions.assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 2  ------------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.70914, 0.90578, 0.75517, 0.20707, 0.85522, 0.70478, 0.77122, 0.4555, 0.84988, 0.98193, 0.96398, 0.28262, 0.51808, 0.07491, 0.34441, 0.65752, 0.12087, 0.75863, 0.9398, 0.32988, 0.03278, 0.99643};
        aRowPointers = new int[]{0, 1, 3, 3, 3, 3, 6, 9, 12, 14, 15, 17, 18, 19, 20, 22};
        aColIndices = new int[]{13, 4, 10, 4, 6, 7, 7, 9, 13, 0, 1, 8, 3, 14, 7, 8, 10, 13, 7, 8, 1, 13};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.80679+0.72595i"), new Complex128("0.64631+0.11629i")},
                {new Complex128("0.89075+0.16705i"), new Complex128("0.71251+0.47285i")},
                {new Complex128("0.3762+0.76103i"), new Complex128("0.05266+0.93714i")},
                {new Complex128("0.07759+0.41809i"), new Complex128("0.80025+0.23838i")},
                {new Complex128("0.91444+0.04866i"), new Complex128("0.97895+0.8661i")},
                {new Complex128("0.89021+0.43522i"), new Complex128("0.33823+0.52676i")},
                {new Complex128("0.7156+0.54231i"), new Complex128("0.67533+0.84657i")},
                {new Complex128("0.31505+0.56928i"), new Complex128("0.99666+0.65255i")},
                {new Complex128("0.29663+0.43686i"), new Complex128("0.64303+0.01951i")},
                {new Complex128("0.74201+0.89189i"), new Complex128("0.57734+0.0887i")},
                {new Complex128("0.22257+0.27625i"), new Complex128("0.36719+0.51197i")},
                {new Complex128("0.69806+0.99721i"), new Complex128("0.99225+0.24712i")},
                {new Complex128("0.98982+0.49989i"), new Complex128("0.74925+0.81972i")},
                {new Complex128("0.87877+0.69179i"), new Complex128("0.163+0.06568i")},
                {new Complex128("0.41477+0.2585i"), new Complex128("0.11814+0.47146i")}};
        B = new CMatrix(bEntries);

        expEntries = new Complex128[][]{
                {new Complex128("0.6231709578+0.4905759606i"), new Complex128("0.11558982000000001+0.0465763152i")},
                {new Complex128("0.9963596501+0.2526909673i"), new Complex128("1.1640042033+1.1711204429i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0233894618+0.8750875428i"), new Complex128("1.4826929339000001+1.3632511113999999i")},
                {new Complex128("1.3278074636000001+1.4332345018i"), new Complex128("1.1701529352000002+0.5994825793999999i")},
                {new Complex128("1.7347100602999999+0.9973303157i"), new Complex128("1.5032097067+0.5755204988999999i")},
                {new Complex128("0.0712682479+0.2359683022i"), new Complex128("0.4234433874+0.158816979i")},
                {new Complex128("0.10850637049999999+0.1960657248i"), new Complex128("0.3432596706+0.22474474549999998i")},
                {new Complex128("0.22194219350000002+0.3206345247i"), new Complex128("0.46718734089999997+0.0747100291i")},
                {new Complex128("0.6666612851000001+0.5248126477i"), new Complex128("0.12365669000000001+0.0498268184i")},
                {new Complex128("0.29608398999999996+0.535009344i"), new Complex128("0.9366610679999999+0.61326649i")},
                {new Complex128("0.0978523044+0.14411137680000002i"), new Complex128("0.2121227364+0.0064359588i")},
                {new Complex128("0.9048315761+0.6947962087i"), new Complex128("0.1857741678+0.08094554540000001i")}};
        exp = new CMatrix(expEntries);

        Assertions.assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 3 ------------------------
        A = new CsrMatrix(24, 516);
        B = new CMatrix(15, 12);
        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }


    @Test
    void standardTransposeTests() {
        // ------------------------ sub-case 1  ------------------------
        aShape = new Shape(5, 5);
        aEntries = new double[]{0.40775, 0.00066, 0.90251, 0.83958, 0.50831, 0.33647, 0.04011, 0.62236};
        aRowPointers = new int[]{0, 3, 5, 5, 7, 8};
        aColIndices = new int[]{1, 3, 4, 0, 4, 1, 4, 3};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.48204+0.59198i"), new Complex128("0.66808+0.35812i"), new Complex128("0.28052+0.3799i"), new Complex128("0.59171+0.66545i"), new Complex128("0.57884+0.73151i")},
                {new Complex128("0.77302+0.60005i"), new Complex128("0.61617+0.36969i"), new Complex128("0.56317+0.19616i"), new Complex128("0.61275+0.4591i"), new Complex128("0.75275+0.95003i")},
                {new Complex128("0.46551+0.54915i"), new Complex128("0.95026+0.3433i"), new Complex128("0.93448+0.43158i"), new Complex128("0.8449+0.8074i"), new Complex128("0.07347+0.0873i")},
                {new Complex128("0.66999+0.13998i"), new Complex128("0.08384+0.55878i"), new Complex128("0.08645+0.38561i"), new Complex128("0.89806+0.23426i"), new Complex128("0.0235+0.29708i")},
                {new Complex128("0.05227+0.40564i"), new Complex128("0.87567+0.41385i"), new Complex128("0.45123+0.28923i"), new Complex128("0.34363+0.6419i"), new Complex128("0.92574+0.99279i")}};
        B = new CMatrix(bEntries).T();

        expEntries = new Complex128[][]{
                {new Complex128("0.3628152961+0.6108569307i"), new Complex128("1.0415995836+0.5246136558000001i"), new Complex128("0.6369292118000001+0.3412717099i"), new Complex128("0.5605710434+0.7666738056000001i"), new Complex128("1.1424389299+1.2835737082i")},
                {new Complex128("0.4312805069+0.7032054368i"), new Complex128("1.0060184241+0.5110344831i"), new Complex128("0.46488370290000003+0.4659749433i"), new Complex128("0.6714584470999999+0.8849827000000001i"), new Complex128("0.9565453866000001+1.1188062507i")},
                {new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.2621945891+0.2181690439i"), new Complex128("0.2424458436+0.1409891178i"), new Complex128("0.20758864519999998+0.07760297050000001i"), new Complex128("0.2199549918+0.180219986i"), new Complex128("0.2904092239+0.35947740100000003i")},
                {new Complex128("0.4169749764+0.0871179528i"), new Complex128("0.0521786624+0.3477623208i"), new Complex128("0.053803022+0.23998823960000001i"), new Complex128("0.5589166216+0.1457940536i"), new Complex128("0.01462546+0.18489070880000003i")}};
        exp = new CMatrix(expEntries);

        Assertions.assertEquals(exp, RealFieldDenseCsrMatMult.standardTranspose(A, B));

        // ------------------------ sub-case 2  ------------------------
        aShape = new Shape(15, 15);
        aEntries = new double[]{0.70914, 0.90578, 0.75517, 0.20707, 0.85522, 0.70478, 0.77122, 0.4555, 0.84988, 0.98193, 0.96398, 0.28262, 0.51808, 0.07491, 0.34441, 0.65752, 0.12087, 0.75863, 0.9398, 0.32988, 0.03278, 0.99643};
        aRowPointers = new int[]{0, 1, 3, 3, 3, 3, 6, 9, 12, 14, 15, 17, 18, 19, 20, 22};
        aColIndices = new int[]{13, 4, 10, 4, 6, 7, 7, 9, 13, 0, 1, 8, 3, 14, 7, 8, 10, 13, 7, 8, 1, 13};
        A = new CsrMatrix(aShape, aEntries, aRowPointers, aColIndices);

        bEntries = new Complex128[][]{
                {new Complex128("0.80679+0.72595i"), new Complex128("0.64631+0.11629i")},
                {new Complex128("0.89075+0.16705i"), new Complex128("0.71251+0.47285i")},
                {new Complex128("0.3762+0.76103i"), new Complex128("0.05266+0.93714i")},
                {new Complex128("0.07759+0.41809i"), new Complex128("0.80025+0.23838i")},
                {new Complex128("0.91444+0.04866i"), new Complex128("0.97895+0.8661i")},
                {new Complex128("0.89021+0.43522i"), new Complex128("0.33823+0.52676i")},
                {new Complex128("0.7156+0.54231i"), new Complex128("0.67533+0.84657i")},
                {new Complex128("0.31505+0.56928i"), new Complex128("0.99666+0.65255i")},
                {new Complex128("0.29663+0.43686i"), new Complex128("0.64303+0.01951i")},
                {new Complex128("0.74201+0.89189i"), new Complex128("0.57734+0.0887i")},
                {new Complex128("0.22257+0.27625i"), new Complex128("0.36719+0.51197i")},
                {new Complex128("0.69806+0.99721i"), new Complex128("0.99225+0.24712i")},
                {new Complex128("0.98982+0.49989i"), new Complex128("0.74925+0.81972i")},
                {new Complex128("0.87877+0.69179i"), new Complex128("0.163+0.06568i")},
                {new Complex128("0.41477+0.2585i"), new Complex128("0.11814+0.47146i")}};
        B = new CMatrix(bEntries).T();

        expEntries = new Complex128[][]{
                {new Complex128("0.6231709578+0.4905759606i"), new Complex128("0.11558982000000001+0.0465763152i")},
                {new Complex128("0.9963596501+0.2526909673i"), new Complex128("1.1640042033+1.1711204429i")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("1.0233894618+0.8750875428i"), new Complex128("1.4826929339000001+1.3632511113999999i")},
                {new Complex128("1.3278074636000001+1.4332345018i"), new Complex128("1.1701529352000002+0.5994825793999999i")},
                {new Complex128("1.7347100602999999+0.9973303157i"), new Complex128("1.5032097067+0.5755204988999999i")},
                {new Complex128("0.0712682479+0.2359683022i"), new Complex128("0.4234433874+0.158816979i")},
                {new Complex128("0.10850637049999999+0.1960657248i"), new Complex128("0.3432596706+0.22474474549999998i")},
                {new Complex128("0.22194219350000002+0.3206345247i"), new Complex128("0.46718734089999997+0.0747100291i")},
                {new Complex128("0.6666612851000001+0.5248126477i"), new Complex128("0.12365669000000001+0.0498268184i")},
                {new Complex128("0.29608398999999996+0.535009344i"), new Complex128("0.9366610679999999+0.61326649i")},
                {new Complex128("0.0978523044+0.14411137680000002i"), new Complex128("0.2121227364+0.0064359588i")},
                {new Complex128("0.9048315761+0.6947962087i"), new Complex128("0.1857741678+0.08094554540000001i")}};
        exp = new CMatrix(expEntries);

        Assertions.assertEquals(exp, RealFieldDenseCsrMatMult.standardTranspose(A, B));

        // ------------------------ sub-case 3 ------------------------
        A = new CsrMatrix(24, 516);
        B = new CMatrix(15, 12).T();
        assertThrows(IllegalArgumentException.class, ()->RealFieldDenseCsrMatMult.standardTranspose(A, B));
    }
}
